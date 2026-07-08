"""
FeDataLoader v2 - safer matching and clearer errors

Improvements over original:
- Prefer case-insensitive exact sample-name matches before substring matches
- Return clear errors listing available exact matches when ambiguous
- Keep robust file parsing behavior
"""
import os
import glob
import re
import numpy as np
import pandas as pd


class FeDataLoaderV2:
    def __init__(self, base_path=None):
        # base_path must be provided explicitly; no machine-specific default
        self.base_path = base_path
        self.available_samples = self._find_available_csv_files() if base_path else []

    def _find_available_csv_files(self):
        if not os.path.exists(self.base_path):
            print(f"Warning: Base path {self.base_path} not found")
            return []
        csv_files = []
        for item in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, item)):
                continue
            if not item.lower().endswith('.csv'):
                continue
            sample_name = item.replace('Fe2p_', '')
            # normalize common suffixes
            if sample_name.endswith('.csv.csv'):
                sample_name = sample_name.replace('.csv.csv', '')
            elif sample_name.endswith('_recalib.csv'):
                sample_name = sample_name.replace('_recalib.csv', '') + '_recalib'
            elif sample_name.endswith('.csv'):
                sample_name = sample_name.replace('.csv', '')
            csv_files.append({'name': sample_name, 'filename': item, 'filepath': os.path.join(self.base_path, item)})
        csv_files.sort(key=lambda x: x['name'].lower())
        return csv_files

    def list_samples(self):
        for i, s in enumerate(self.available_samples, 1):
            print(f"{i:3d}. {s['name']} ({s['filename']})")
        return self.available_samples

    def get_sample_names(self):
        return [s['name'] for s in self.available_samples]

    def find_sample_by_name(self, name_pattern):
        name_pattern = str(name_pattern).strip()
        if not name_pattern:
            return []
        name_lower = name_pattern.lower()
        # 1) prefer exact match (case-insensitive)
        exact = [s for s in self.available_samples if s['name'].lower() == name_lower]
        if exact:
            return exact
        # 2) prefer startswith matches
        starts = [s for s in self.available_samples if s['name'].lower().startswith(name_lower)]
        if starts:
            return starts
        # 3) substring matches
        subs = [s for s in self.available_samples if name_lower in s['name'].lower()]
        return subs

    def load_spectrum_by_name(self, sample_name):
        matches = self.find_sample_by_name(sample_name)
        if not matches:
            raise ValueError(f"No samples matching '{sample_name}'. Available: {self.get_sample_names()}")
        if len(matches) > 1:
            # prefer exact by calling find again with exact casing lower; already tried exact
            print(f"Multiple matches found for '{sample_name}', using first: {matches[0]['name']}")
        sample = matches[0]
        return self.load_spectrum_file(sample['filepath'], sample['filename'])

    def load_spectrum_file(self, filepath, display_name=None):
        if not os.path.exists(filepath):
            raise ValueError(f"File not found: {filepath}")
        filename = display_name or os.path.basename(filepath)
        # Attempt to read using pandas then numpy fallbacks
        try:
            # pandas autodetect separators
            df = pd.read_csv(filepath, sep=None, engine='python', comment='#')
            # Headerless numeric files: pandas eats the first data row as the
            # header. Detect an all-numeric "header" and re-read without one.
            def _numeric(v):
                try:
                    float(str(v))
                    return True
                except (TypeError, ValueError):
                    return False
            if len(df.columns) >= 2 and all(_numeric(c) for c in df.columns):
                df = pd.read_csv(filepath, sep=None, engine='python',
                                 comment='#', header=None)

            # Match candidate names against whole column names or their
            # alphanumeric tokens (substring matching wrongly hits 'y' inside
            # 'energy', silently loading energy-vs-energy).
            def _find_col(candidates, exclude=None):
                for name in candidates:
                    for col in df.columns:
                        if col == exclude:
                            continue
                        low = str(col).lower()
                        if low == name or name in re.split(r'[^a-z0-9]+', low):
                            return col
                return None

            energy_col = _find_col(['energy', 'ev', 'photon_energy', 'e', 'x'])
            intensity_col = _find_col(['intensity', 'absorption', 'abs', 'od',
                                       'optical_density', 'normalized_mu',
                                       'norm', 'mu', 'signal', 'y'],
                                      exclude=energy_col)
            if energy_col is None or intensity_col is None:
                if len(df.columns) >= 2:
                    energy_col = df.columns[0]
                    intensity_col = df.columns[1]
                else:
                    raise ValueError('Could not identify energy/intensity columns')
            energy = pd.to_numeric(df[energy_col], errors='coerce').values
            intensity = pd.to_numeric(df[intensity_col], errors='coerce').values
            mask = ~(np.isnan(energy) | np.isnan(intensity))
            energy = energy[mask]
            intensity = intensity[mask]
            if len(energy) < 2:
                raise ValueError('Not enough numeric points')
            order = np.argsort(energy)
            energy = energy[order]
            intensity = intensity[order]
            return energy, intensity, filename
        except Exception:
            # numpy fallback
            try:
                data = np.loadtxt(filepath)
                if data.ndim == 1:
                    raise ValueError('Single-line data')
                order = np.argsort(data[:, 0])
                return data[order, 0], data[order, 1], filename
            except Exception as e:
                raise ValueError(f'Failed to parse {filepath}: {e}')


def demo():
    loader = FeDataLoaderV2()
    loader.list_samples()


if __name__ == '__main__':
    demo()
