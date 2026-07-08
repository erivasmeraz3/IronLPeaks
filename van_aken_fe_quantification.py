#!/usr/bin/env python3
"""
van Aken & Liebscher (2002) Fe3+/SFe quantification from Fe L2,3-edge spectra.

Modified integral white-line intensity ratio method:
  - Two ~2-eV integration windows isolate Fe3+-sensitive features
  - L3 window: 708.5 - 710.5 eV (Fe3+ component of L3 edge)
  - L2 window: 719.7 - 721.7 eV (Fe2+ component of L2 edge)
  - p = I(L2') / (I(L2') + I(L3'))
  - p(x) = a*x^2 + b*x + c,  where x = Fe3+/SFe
    a = 0.193, b = -0.465, c = 0.366  (r^2 = 0.9985)
  - Absolute errors: +/-0.03 to +/-0.04 for Fe3+/SFe

Also computes the L3 centroid energy (intensity-weighted mean energy of the
L3 white line), which shifts from ~707.8 eV (Fe2+) to ~709.5 eV (Fe3+) and
serves as an independent qualitative indicator.

References:
  van Aken, P.A. & Liebscher, B. (2002) Quantification of ferrous/ferric
  ratios in minerals: new evaluation schemes of Fe L23 electron energy-loss
  near-edge spectra. Phys Chem Minerals 29:188-200.

  van Aken, P.A., Liebscher, B. & Styrsa, V.J. (1998) Quantitative
  determination of iron oxidation states in minerals using Fe L2,3-edge
  electron energy-loss near-edge structure spectroscopy. Phys Chem Minerals
  25:323-327.

Note: Calibration was performed on high-spin Fe bonded to oxygen (garnets,
spinels, pyroxenes, oxides, hydroxides, olivines). Fe-S bonds (e.g.
mackinawite) may introduce systematic offsets.
"""

import numpy as np


class VanAkenFeQuantifier:
    """van Aken & Liebscher (2002) Fe3+/SFe from Fe L2,3-edge spectra."""

    # -- Modified integral method: integration windows (eV) --
    DEFAULT_L3_WINDOW = (708.5, 710.5)   # L3' window (Fe3+ component)
    DEFAULT_L2_WINDOW = (719.7, 721.7)   # L2' window

    # -- Universal curve coefficients (Eq. 2, van Aken & Liebscher 2002) --
    # p(x) = a*x^2 + b*x + c  where p = I(L2')/(I(L2')+I(L3')), x = Fe3+/SFe
    CALIB_A = 0.193
    CALIB_B = -0.465
    CALIB_C = 0.366

    # -- L3 centroid range --
    DEFAULT_L3_CENTROID_RANGE = (705.0, 715.0)

    def __init__(self, energy, intensity,
                 l3_window=None, l2_window=None,
                 l3_centroid_range=None):
        """
        Parameters
        ----------
        energy : array-like
            Energy axis (eV) of the baseline-corrected spectrum.
        intensity : array-like
            Baseline-corrected intensity values.
        l3_window : tuple(float, float) or None
            L3' integration window (default 708.5-710.5 eV).
        l2_window : tuple(float, float) or None
            L2' integration window (default 719.7-721.7 eV).
        l3_centroid_range : tuple(float, float) or None
            Energy range for L3 centroid calculation (default 705-715 eV).
        """
        self.energy = np.asarray(energy, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        if self.energy.size and np.any(np.diff(self.energy) < 0):
            order = np.argsort(self.energy)
            self.energy = self.energy[order]
            self.intensity = self.intensity[order]

        self.l3_window = l3_window if l3_window is not None else self.DEFAULT_L3_WINDOW
        self.l2_window = l2_window if l2_window is not None else self.DEFAULT_L2_WINDOW
        self.l3_centroid_range = (l3_centroid_range if l3_centroid_range is not None
                                  else self.DEFAULT_L3_CENTROID_RANGE)

        # Results
        self.il3 = None          # Integrated L3' window intensity
        self.il2 = None          # Integrated L2' window intensity
        self.ratio = None        # I(L3')/I(L2') ratio
        self.p_value = None      # I(L2')/(I(L2')+I(L3'))
        self.fe3 = None          # Fe3+/SFe from modified integral method
        self.l3_centroid = None  # L3 centroid energy (eV)

    @staticmethod
    def _solve_quadratic(p, a=None, b=None, c=None):
        """Solve a*x^2 + b*x + (c - p) = 0 for x in [0, 1].

        Returns Fe3+/SFe (clamped to [0,1]) or NaN if no valid root.
        """
        if a is None:
            a = VanAkenFeQuantifier.CALIB_A
        if b is None:
            b = VanAkenFeQuantifier.CALIB_B
        if c is None:
            c = VanAkenFeQuantifier.CALIB_C

        # a*x^2 + b*x + (c - p) = 0
        discriminant = b**2 - 4.0 * a * (c - p)
        if discriminant < 0:
            return np.nan

        sqrt_d = np.sqrt(discriminant)
        x1 = (-b + sqrt_d) / (2.0 * a)
        x2 = (-b - sqrt_d) / (2.0 * a)

        # Pick the root in [0, 1]; prefer the one closest to [0, 1]
        candidates = []
        for x in (x1, x2):
            if -0.05 <= x <= 1.05:  # small tolerance
                candidates.append(x)

        if not candidates:
            return np.nan

        # Return the root closest to the valid range center
        best = min(candidates, key=lambda v: abs(v - 0.5))
        return float(np.clip(best, 0.0, 1.0))

    def _window_integral(self, lo, hi):
        """Trapezoid integral over [lo, hi] with the intensity interpolated
        onto the exact window boundaries.

        Truncating to interior grid points drops up to one grid step of the
        2 eV window at each edge; when the two windows are sampled at
        different densities (common in STXM stacks with fine L3 / coarse L2
        regions) that truncation is asymmetric and biases p. Requires the
        data to fully cover the window (partial coverage returns NaN).
        """
        e, i = self.energy, self.intensity
        if lo >= hi or len(e) < 2 or e[0] > lo or e[-1] < hi:
            return np.nan
        mask = (e > lo) & (e < hi)
        xs = np.concatenate([[lo], e[mask], [hi]])
        ys = np.concatenate([[np.interp(lo, e, i)], i[mask],
                             [np.interp(hi, e, i)]])
        return float(np.trapezoid(ys, xs))

    def compute_modified_integral(self):
        """Modified integral white-line intensity ratio method.

        Integrates baseline-corrected spectrum in the L3' and L2' windows,
        computes p = I(L2')/(I(L2')+I(L3')), and solves the universal curve
        polynomial for Fe3+/SFe.

        Returns
        -------
        float
            Fe3+/SFe (clamped to [0, 1]) or NaN.
        """
        self.il3 = self._window_integral(*self.l3_window)

        if not np.isfinite(self.il3):
            self.il3 = np.nan
            self.il2 = np.nan
            self.ratio = np.nan
            self.p_value = np.nan
            self.fe3 = np.nan
            return self.fe3

        self.il2 = self._window_integral(*self.l2_window)

        if not np.isfinite(self.il2):
            self.il2 = np.nan
            self.ratio = np.nan
            self.p_value = np.nan
            self.fe3 = np.nan
            return self.fe3

        # Compute ratios
        total = self.il3 + self.il2
        if total <= 0:
            self.ratio = np.nan
            self.p_value = np.nan
            self.fe3 = np.nan
            return self.fe3

        self.ratio = self.il3 / self.il2 if self.il2 > 0 else np.nan
        self.p_value = self.il2 / total

        # Solve universal curve for Fe3+/SFe
        self.fe3 = self._solve_quadratic(self.p_value)
        return self.fe3

    def compute_l3_centroid(self):
        """Compute L3 centroid energy (intensity-weighted mean).

        The centroid shifts from ~707.8 eV (pure Fe2+) to ~709.5 eV
        (pure Fe3+).

        Returns
        -------
        float
            L3 centroid energy in eV, or NaN.
        """
        lo, hi = self.l3_centroid_range
        mask = (self.energy >= lo) & (self.energy <= hi)
        e = self.energy[mask]
        i = self.intensity[mask]

        # Only use positive intensities for centroid
        pos = i > 0
        if np.sum(pos) < 2:
            self.l3_centroid = np.nan
            return self.l3_centroid

        e_pos = e[pos]
        i_pos = i[pos]
        self.l3_centroid = float(np.sum(e_pos * i_pos) / np.sum(i_pos))
        return self.l3_centroid

    def compute_all(self):
        """Run modified integral method and L3 centroid.

        Returns
        -------
        dict
            All computed values.
        """
        self.compute_modified_integral()
        self.compute_l3_centroid()
        return self.get_results()

    def get_results(self):
        """Return dict with all computed values.

        Returns
        -------
        dict
            Keys: fe3, ratio, p_value, il3, il2, l3_window, l2_window,
                  l3_centroid
        """
        return {
            'fe3': self.fe3,
            'ratio': self.ratio,
            'p_value': self.p_value,
            'il3': self.il3,
            'il2': self.il2,
            'l3_lo': self.l3_window[0],
            'l3_hi': self.l3_window[1],
            'l2_lo': self.l2_window[0],
            'l2_hi': self.l2_window[1],
            'l3_centroid': self.l3_centroid,
        }
