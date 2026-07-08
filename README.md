# IronLPeaks: Fe L-edge XANES Analysis Software

**Version 1.0**

IronLPeaks is a desktop application for analyzing iron L2,3-edge X-ray absorption spectra from STXM and related spectromicroscopy techniques. It fits a double-arctangent continuum beneath the Fe L2,3 white lines, resolves the L3 white line into sub-peaks, and quantifies the ferric iron fraction Fe3+/&Sigma;Fe with the van Aken and Liebscher (2002) modified integral method. The fitted L3 sub-peaks also provide the peak splitting descriptors (energy separation and intensity ratio) used to distinguish Fe-bearing phases (von der Heyden et al., 2017).

Companion tools by the same author: [CarbonKPeaks](https://github.com/erivasmeraz3/CarbonKPeaks) for C 1s spectra and [SulfurKPeaks](https://github.com/erivasmeraz3/SulfurKPeaks) for S K-edge spectra.

## Features

- Interactive Tkinter + Matplotlib GUI: peak clicking, baseline overlay, van Aken integration windows
- Double-arctangent continuum fit with an enforced 2:1 L3:L2 branching ratio
- Automatic second-derivative (Savitzky-Golay) peak detection in the L3 region
- Fe3+/&Sigma;Fe quantification by the van Aken and Liebscher (2002) modified integral white-line method
- L3 peak splitting outputs (delta-E and intensity ratio) for phase discrimination
- Sample classification, batch processing, and CSV export
- Athena .prj project import

## Installation

### Run from source

```bash
pip install -r requirements.txt   # tkinter ships with Python
python peak_selector_gui_v2.py
```

### Build a standalone Windows executable

```bash
python build.py
# Output: dist/IronLPeaks.exe
```

## Quick Start

1. **Load** spectra with the file dialog, or start the GUI pointed at a folder: `python peak_selector_gui_v2.py --spectra-dir /path/to/csv/files`. Athena projects can be imported directly.
2. **Select** a sample from the list. The double-arctangent baseline is fitted and drawn under the spectrum.
3. **Pick peaks** by accepting the automatically detected L3 sub-peaks or clicking peak positions on the plot.
4. **Quantify**: the L3' (708.5 to 710.5 eV) and L2' (719.7 to 721.7 eV) windows are integrated above the continuum and the calibration polynomial returns Fe3+/&Sigma;Fe.
5. **Export** batch results (Fe3+/&Sigma;Fe, L3 peak positions, splitting, intensity ratio, quality flags) to CSV.

## Input Data Format

Two-column normalized Fe L-edge spectra with energy in eV (approximately 695 to 740). CSV column names are auto-detected. Athena `.prj` projects are supported in the GUI, and `extract_athena_spectra.py` can also export them to CSV from the command line:

```bash
python extract_athena_spectra.py project.prj --output spectra/ --emin 695 --emax 740
```

## Method Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| L3' integration window | 708.5 to 710.5 eV | van Aken and Liebscher (2002) |
| L2' integration window | 719.7 to 721.7 eV | van Aken and Liebscher (2002) |
| Calibration polynomial a, b, c | 0.193, -0.465, 0.366 | van Aken and Liebscher (2002), Eq. 2 |
| L3:L2 branching ratio | 2:1 | van Aken (2002); Calvert et al. (2005) |
| Continuum edge positions, width | 708.65 eV, 721.65 eV, w = 1 eV (fixed) | van Aken and Liebscher (2002), Eq. 1 |
| L3 sub-peak separation limits | 0.8 to 3.5 eV | physical constraint |

The calibration uses p(x) = 0.193 x^2 - 0.465 x + 0.366, where p = I(L2') / (I(L2') + I(L3')) and x = Fe3+/&Sigma;Fe.

## References

- van Aken, P.A. and Liebscher, B. (2002) Quantification of ferrous/ferric ratios in minerals: new evaluation schemes of Fe L23 electron energy-loss near-edge spectra. Physics and Chemistry of Minerals 29, 188-200.
- von der Heyden, B.P. et al. (2017) American Mineralogist 102, 674.
- Calvert, C.C. et al. (2005) Journal of Electron Spectroscopy and Related Phenomena 143, 173-187.
