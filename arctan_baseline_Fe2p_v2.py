"""
ArctanBaselineFe2p v2
- quieter by default (use verbose=True to enable prints)
- returns structured result dict and raises on unrecoverable errors
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import minimum_filter1d


class ArctanBaselineFe2pV2:
    def __init__(self, energy, intensity, verbose=False,
                 penalty_weight=200.0,
                 enable_iterative_refit=True,
                 max_refit_attempts=10,
                 refit_shift=0.2,
                 step_reduction_factor=0.9):
        self.energy = np.array(energy)
        self.intensity = np.array(intensity)
        self.energy_range = float(self.energy.max() - self.energy.min())
        self.verbose = verbose
        self.baseline = None
        self.intensity_corrected = None
        self.derivative = None
        self.baseline_params = None
        self._set_analysis_regions()
        # Configurable behavior for preventing baseline pre-edge crossing
        # penalty_weight: strength of soft penalty applied when baseline > data
        #   suggested range: 0 (off) | 50..500 (moderate) | 500..2000 (strong)
        # enable_iterative_refit: if True, perform hybrid iterative refits when
        #   post-fit violations are detected
        # max_refit_attempts: up to this many refit attempts (suggested 3..20)
        # refit_shift: unused (edge positions are fixed at the published van
        #   Aken values); retained for API compatibility
        # step_reduction_factor: multiplicative factor to reduce allowed step per retry (0.5..0.95)
        self.penalty_weight = float(penalty_weight)
        self.enable_iterative_refit = bool(enable_iterative_refit)
        self.max_refit_attempts = int(max_refit_attempts)
        self.refit_shift = float(refit_shift)
        self.step_reduction_factor = float(step_reduction_factor)

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _set_analysis_regions(self):
        if (self.energy.min() > 700.5 and self.energy.max() < 741 and len(self.energy) < 70):
            self.pre_edge_range = (700.0, 703.0)
            self.mid_edge_range = (715, 719)
        elif self.energy_range < 35:
            self.pre_edge_range = (max(690, self.energy.min()), min(705, self.energy.min() + 10))
            self.mid_edge_range = (715, 719)
            if self.energy.min() > 699 and self.energy_range < 32:
                self.pre_edge_range = (self.energy.min(), min(704, self.energy.min() + 4))
        else:
            self.pre_edge_range = (690, 700)
            self.mid_edge_range = (715, 719)
        self.post_edge_range = (730, min(750, self.energy.max()))

    @staticmethod
    def arctan_baseline(energy, offset, slope, step, width, edge_pos):
        linear = offset + slope * energy
        arctan_component = (step / np.pi) * np.arctan((energy - edge_pos) / width)
        return linear + arctan_component

    @staticmethod
    def double_arctan_baseline(energy, offset, slope, step, width, edge_l3, edge_l2):
        """Double arctangent baseline (Calvert et al. 2005).

        L3:L2 branching ratio fixed at 2:1.
        """
        linear = offset + slope * energy
        l3_component = (2 * step / (3 * np.pi)) * np.arctan((energy - edge_l3) / width)
        l2_component = (step / (3 * np.pi)) * np.arctan((energy - edge_l2) / width)
        return linear + l3_component + l2_component

    def _detect_l3_rising_edge(self, baseline_level):
        l3_mask = (self.energy >= 704.0) & (self.energy <= 709.0)
        if not np.any(l3_mask):
            return 705.5
        e_l3 = self.energy[l3_mask]
        i_l3 = self.intensity[l3_mask]
        if len(i_l3) < 3:
            return 705.5
        std = np.std(i_l3)
        thr = max(0.005, std * 0.5)
        rising = i_l3 > (baseline_level + thr)
        if np.any(rising):
            idx = np.where(rising)[0][0]
            val = e_l3[idx]
            if 704.5 <= val <= 708.0:
                return val
        # rolling min
        if len(i_l3) >= 5:
            roll = minimum_filter1d(i_l3, size=3)
            for i in range(1, len(i_l3)):
                if (i_l3[i] > roll[i] + 0.008) and (i_l3[i] > i_l3[i-1]):
                    val = e_l3[i]
                    if 704.5 <= val <= 708.0:
                        return val
        # derivative
        if len(i_l3) > 6:
            deriv = np.gradient(i_l3, e_l3)
            window = min(5, len(deriv))
            if window >= 3:
                smooth = np.convolve(deriv, np.ones(window)/window, mode='same')
                dthr = max(0.003, np.std(smooth) * 0.3)
                for i in range(1, len(smooth)):
                    if (smooth[i] > dthr) and (smooth[i] > smooth[i-1]):
                        val = e_l3[i]
                        if 704.5 <= val <= 708.0:
                            return val
        return 705.5

    def fit_baseline(self, pre_edge_range=None, mid_edge_range=None, post_edge_range=None, sample_name='Unknown'):
        self.sample_name = sample_name
        pre_range = pre_edge_range or self.pre_edge_range
        mid_range = mid_edge_range or self.mid_edge_range
        post_range = post_edge_range or self.post_edge_range
        try:
            pre_mask = (self.energy >= pre_range[0]) & (self.energy <= pre_range[1])
            mid_mask = (self.energy >= mid_range[0]) & (self.energy <= mid_range[1])
            if pre_mask.sum() < 2:
                broader = (self.energy >= max(self.energy.min(), 700.0)) & (self.energy <= min(self.energy.max(), 705.0))
                if broader.sum() >= 2:
                    pre_mask = broader
                else:
                    raise ValueError('Insufficient pre-edge points')
            if mid_mask.sum() < 2:
                post_l3 = (self.energy >= 710.0) & (self.energy <= min(self.energy.max(), 720.0))
                if post_l3.sum() >= 2:
                    mid_mask = post_l3
                else:
                    raise ValueError('Insufficient mid-edge points')
            pre_e = self.energy[pre_mask]; pre_i = self.intensity[pre_mask]
            mid_e = self.energy[mid_mask]; mid_i = self.intensity[mid_mask]
            pre_idx = np.argmin(pre_i); mid_idx = np.argmin(mid_i)
            pre_anchor_e = pre_e[pre_idx]; pre_anchor_l = pre_i[pre_idx]
            mid_anchor_e = mid_e[mid_idx]; mid_anchor_l = mid_i[mid_idx]
            step_mask = (self.energy >= 715.0) & (self.energy <= 719.0)
            if np.any(step_mask):
                min_715 = np.min(self.intensity[step_mask])
            else:
                min_715 = np.min(self.intensity[self.energy > 715.0]) if np.any(self.energy > 715.0) else np.min(self.intensity)
            l3_edge = self._detect_l3_rising_edge(pre_anchor_l)
            initial_step = max(0.01, mid_anchor_l - pre_anchor_l)
            # Only the L3 component (2/3 of the total double-arctan step) has
            # risen at the 715-719 eV valley, so keeping the baseline at or
            # below the valley requires step <= 1.5*(valley - pre-edge).
            # Capping at 1.0x excludes the true continuum by construction.
            max_allowed_step = 1.5 * (min_715 - pre_anchor_l)
            if max_allowed_step <= 0:
                max_allowed_step = 0.05
            initial_step = min(initial_step, max_allowed_step * 0.8)
            if initial_step <= 0:
                initial_step = 0.001
            # The van Aken & Liebscher (2002) calibration polynomial was
            # derived with a FIXED continuum shape: edge onsets E_L3 = 708.65
            # and E_L2 = 721.65 eV with width w = 1 eV (their Eq. 1; in this
            # parametrization the arctan scale is w/pi = 0.318 eV). The white
            # lines are excluded from the data term below, so these shape
            # parameters are invisible to the data; left free they slide to
            # whatever bound eats white-line intensity (an early/narrow L2
            # edge subtracts real L2' signal and overestimates Fe3+/SFe).
            # Since the L3'/L2' integration windows are fixed absolute
            # energies, applying the calibration faithfully means subtracting
            # the SAME fixed continuum shape: fit only offset, slope, step.
            FIXED_WIDTH = 1.0 / np.pi     # van Aken w = 1 eV
            FIXED_EDGE_L3 = 708.65
            FIXED_EDGE_L2 = 721.65

            def full_params(p3):
                return [p3[0], p3[1], p3[2],
                        FIXED_WIDTH, FIXED_EDGE_L3, FIXED_EDGE_L2]

            p0 = [pre_anchor_l, 0.001, initial_step]
            lower = [-np.inf, -0.02, 0.0]
            upper = [np.inf, 0.02, max_allowed_step]

            anchor_energies = [pre_anchor_e, mid_anchor_e]
            anchor_levels = [pre_anchor_l, mid_anchor_l]

            # Pre-edge mask for penalties
            pre_mask_full = (self.energy >= pre_range[0]) & (self.energy <= pre_range[1])
            if not np.any(pre_mask_full):
                pre_mask_full = (self.energy >= max(self.energy.min(), 700.0)) & (self.energy <= min(self.energy.max(), 705.0))

            # Post-edge mask for penalties — baseline must stay below
            # the data after the L2 edge to prevent negative corrected intensity
            post_edge_start = max(725.0, post_range[0])
            post_mask_full = (self.energy >= post_edge_start) & (self.energy <= self.energy.max())
            if not np.any(post_mask_full):
                post_mask_full = self.energy >= (self.energy.max() - 5.0)

            # Determine a robust signal scale for normalizing penalty terms
            signal_scale = max(1e-6, np.median(np.abs(self.intensity)))

            # Continuum-only data term. Fitting the residual over the entire
            # spectrum lets the L3/L2 white lines (several times the step
            # height) dominate the least-squares cost and pull the continuum
            # up into the peaks, driving step/edge/width to their bounds.
            # Restrict the data term to regions that ARE continuum: the
            # pre-edge, the L3->L2 valley, and the post-L2 region.
            valley_mask = (self.energy >= 715.0) & (self.energy <= 718.5)
            post_cont_mask = self.energy >= 726.0
            fit_mask = pre_mask_full | valley_mask | post_cont_mask
            if fit_mask.sum() < 8:
                fit_mask = np.ones(len(self.energy), dtype=bool)

            def objective(params):
                fp = full_params(params)
                b = self.double_arctan_baseline(self.energy, *fp)
                res = (self.intensity - b)[fit_mask]
                # Anchor residuals (strong constraints to match anchor minima)
                w = 20.0
                for ae, al in zip(anchor_energies, anchor_levels):
                    anchor_b = self.double_arctan_baseline(np.array([ae]), *fp)[0]
                    res = np.append(res, (al - anchor_b) * w)

                # Soft penalty: where baseline sits above measured pre-edge data,
                # add positive residuals proportional to the exceedance. This
                # discourages the optimizer from putting the arctan step before
                # the rising edge (which creates the negative area in the 2nd deriv).
                if self.penalty_weight > 0 and np.any(pre_mask_full):
                    violations = b[pre_mask_full] - self.intensity[pre_mask_full]
                    pos_viol = np.clip(violations, 0.0, None)
                    if pos_viol.size > 0:
                        # normalize by signal scale so weight is stable across spectra
                        penalty = (self.penalty_weight * pos_viol) / (signal_scale + 1e-12)
                        res = np.concatenate([res, penalty])

                # Post-edge penalty: where baseline sits above measured
                # post-edge data, penalize similarly. This prevents the
                # L2 arctan step from overshooting the post-edge continuum,
                # which would produce negative corrected intensity and
                # deflate the L2' integral.
                if self.penalty_weight > 0 and np.any(post_mask_full):
                    violations = b[post_mask_full] - self.intensity[post_mask_full]
                    pos_viol = np.clip(violations, 0.0, None)
                    if pos_viol.size > 0:
                        penalty = (self.penalty_weight * pos_viol) / (signal_scale + 1e-12)
                        res = np.concatenate([res, penalty])

                return res

            # Initial fit (3 free params; shape fixed at published values)
            res = least_squares(objective, p0, bounds=(lower, upper), max_nfev=5000, method='trf')
            popt3 = res.x
            popt = np.array(full_params(popt3), dtype=float)

            # Hybrid iterative refit: if post-fit violations remain (pre-edge
            # or post-edge), we perform up to `self.max_refit_attempts` retries.
            # Each retry reduces the allowed step amplitude by
            # `step_reduction_factor`. This, combined with the soft penalties,
            # gives robust protection against baseline crossovers.
            pre_mask_full = (self.energy >= pre_range[0]) & (self.energy <= pre_range[1])
            if not np.any(pre_mask_full):
                pre_mask_full = (self.energy >= max(self.energy.min(), 700.0)) & (self.energy <= min(self.energy.max(), 705.0))

            def _max_violation(p):
                bl = self.double_arctan_baseline(self.energy, *p)
                pre_v = bl[pre_mask_full] - self.intensity[pre_mask_full]
                post_v = bl[post_mask_full] - self.intensity[post_mask_full]
                mp = float(np.max(pre_v)) if pre_v.size > 0 else 0.0
                mq = float(np.max(post_v)) if post_v.size > 0 else 0.0
                return max(mp, mq), mp, mq

            attempt = 0
            # Noise-aware tolerance: a least-squares baseline necessarily sits
            # ~2-3 sigma above the lowest noise points, so pure noise
            # excursions must not trigger refits (with a noise-blind tolerance
            # the loop can never converge and each retry shrinks the step
            # bound without rollback, corrupting the continuum).
            base0 = self.double_arctan_baseline(self.energy, *popt)
            resid0 = (self.intensity - base0)[fit_mask]
            sigma = (1.4826 * np.median(np.abs(resid0 - np.median(resid0)))
                     if resid0.size else 0.0)
            tol = max(1e-4,
                      0.002 * (np.median(np.abs(self.intensity)) + 1e-12),
                      2.5 * sigma)
            best_popt = np.array(popt, dtype=float).copy()
            best_violation = np.inf
            while attempt < (self.max_refit_attempts if self.enable_iterative_refit else 0):
                max_violation, max_pre, max_post = _max_violation(popt)
                if max_violation < best_violation:
                    best_violation = max_violation
                    best_popt = np.array(popt, dtype=float).copy()
                self._vprint(f'Hybrid post-fit check attempt {attempt+1}: max pre-edge={max_pre:.6f}, max post-edge={max_post:.6f}, tol={tol:.6f}')
                if max_violation <= tol:
                    break

                # Adjust bounds for next attempt: reduce allowed step
                # amplitude (edge positions/width are fixed, not fitted)
                lower = list(lower)
                upper = list(upper)
                upper[2] = max(0.0, min(upper[2], popt3[2] * self.step_reduction_factor))

                # Prepare new starting point clamped into bounds
                p0 = list(popt3)
                for i in range(len(p0)):
                    if p0[i] < lower[i]:
                        p0[i] = lower[i] + 1e-8
                    if p0[i] > upper[i]:
                        p0[i] = upper[i] - 1e-8

                try:
                    res = least_squares(objective, p0, bounds=(tuple(lower), tuple(upper)), max_nfev=5000, method='trf')
                    popt3 = res.x
                    popt = np.array(full_params(popt3), dtype=float)
                except Exception as e:
                    self._vprint(f'Refit attempt failed: {e}')
                    break

                attempt += 1

            # A refit may never leave us worse than the best fit seen
            final_violation, _, _ = _max_violation(popt)
            if final_violation < best_violation:
                best_popt = np.array(popt, dtype=float).copy()
            popt = best_popt

            # Finalize
            self.baseline = self.double_arctan_baseline(self.energy, *popt)
            self.intensity_corrected = self.intensity - self.baseline
            self.baseline_params = popt
            self.derivative = np.gradient(self.intensity_corrected, self.energy)
            result = {'success': True, 'baseline_params': popt, 'step_height': popt[2], 'edge_position': popt[4], 'edge_l2_position': popt[5], 'edge_width': popt[3], 'post_fit_attempts': attempt}
            return result
        except Exception as e:
            # fallback linear baseline
            try:
                pre_mask = (self.energy >= pre_range[0]) & (self.energy <= pre_range[1])
                post_mask = (self.energy >= post_range[0]) & (self.energy <= post_range[1])
                if np.any(pre_mask) and np.any(post_mask):
                    pre_mean = np.mean(self.intensity[pre_mask]); post_mean = np.mean(self.intensity[post_mask])
                    pre_e = np.mean(self.energy[pre_mask]); post_e = np.mean(self.energy[post_mask])
                    slope = (post_mean - pre_mean) / (post_e - pre_e)
                    offset = pre_mean - slope * pre_e
                    self.baseline = offset + slope * self.energy
                    self.intensity_corrected = self.intensity - self.baseline
                    self.derivative = np.gradient(self.intensity_corrected, self.energy)
                    return {'success': False, 'fallback': True, 'error': str(e)}
                else:
                    raise
            except Exception:
                raise

    def plot_analysis(self, figsize=(10, 8), save_path=None):
        import matplotlib.pyplot as plt
        if self.baseline is None:
            raise ValueError('No baseline fitted')
        fig, axs = plt.subplots(3, 1, figsize=figsize)

        # Normalize by the max absolute original intensity for stable plotting
        max_int = max(1e-12, np.max(np.abs(self.intensity)))
        orig_norm = self.intensity / max_int
        baseline_norm = self.baseline / max_int
        corrected_norm = self.intensity_corrected / max_int

        # Top: normalized original spectrum with baseline overlay
        axs[0].plot(self.energy, orig_norm, label='original (normalized)', color='C0')
        axs[0].plot(self.energy, baseline_norm, '--', label='baseline (normalized)', color='C1')
        axs[0].set_ylabel('Normalized intensity')
        axs[0].set_title(f'{getattr(self, "sample_name", "sample")} — Original + Baseline')
        axs[0].legend(fontsize=8)
        axs[0].grid(True, alpha=0.3)

        # Middle: baseline-subtracted (corrected) normalized
        axs[1].plot(self.energy, corrected_norm, label='baseline-subtracted (normalized)', color='C2')
        axs[1].axhline(0, color='k', linestyle='--', alpha=0.4)
        axs[1].set_ylabel('Corrected (norm)')
        axs[1].set_title('Baseline-subtracted spectrum (normalized)')
        axs[1].legend(fontsize=8)
        axs[1].grid(True, alpha=0.3)

        # Bottom: derivative (computed from corrected intensity) — show both raw and a smoothed version
        try:
            deriv = np.gradient(self.intensity_corrected, self.energy)
            from scipy.ndimage import gaussian_filter1d
            deriv_smooth = gaussian_filter1d(deriv, sigma=1.0)
            axs[2].plot(self.energy, deriv, color='purple', alpha=0.6, label='dI/dE (raw)')
            axs[2].plot(self.energy, deriv_smooth, color='orange', linewidth=1.5, label='dI/dE (smoothed)')
        except Exception:
            axs[2].plot(self.energy, self.derivative, label='derivative')

        axs[2].set_xlabel('Energy (eV)')
        axs[2].set_ylabel('dI/dE')
        axs[2].set_title('Derivative (from baseline-subtracted data)')
        axs[2].legend(fontsize=8)
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig

    def get_baseline_corrected_data(self):
        if self.intensity_corrected is None:
            raise ValueError('No correction applied')
        return self.energy, self.intensity_corrected

    def get_derivative_data(self):
        if self.derivative is None:
            raise ValueError('No derivative computed')
        return self.energy, self.derivative


if __name__ == '__main__':
    print('ArctanBaselineFe2pV2 module')
