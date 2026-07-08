"""
Build script for IronLPeaks application.
Creates a standalone executable with embedded icon.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def install_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller', '-q'])


def build():
    """Build the executable."""
    install_pyinstaller()

    base_dir = Path(__file__).parent
    main_script = base_dir / "peak_selector_gui_v2.py"
    icon_file = base_dir / "ironlpeaks.ico"

    # Generate icon if missing
    if not icon_file.exists():
        print("Icon not found, generating...")
        subprocess.check_call([sys.executable, str(base_dir / "create_icon.py")])

    # Splash screen disabled: the pyi_splash bootloader (its own tcl
    # interpreter) crashes the windowed exe on startup with an access
    # violation under PyInstaller 6.16 / Python 3.13.
    splash_image = None

    # Data files to include (icon PNGs for runtime)
    data_files = []
    for png in base_dir.glob("ironlpeaks_*.png"):
        data_files.append(f"--add-data={png};.")
    if icon_file.exists():
        data_files.append(f"--add-data={icon_file};.")

    # Include companion modules
    companion_modules = [
        "fe_data_loader_v2.py",
        "arctan_baseline_Fe2p_v2.py",
        "van_aken_fe_quantification.py",
        "extract_athena_spectra.py",
    ]
    for mod in companion_modules:
        mod_path = base_dir / mod
        if mod_path.exists():
            data_files.append(f"--add-data={mod_path};.")

    # Exclude heavy packages that live in the dev environment but are not
    # used by IronLPeaks (pulled in via pandas/matplotlib optional-dep hooks;
    # without these the one-file exe balloons from ~90 MB to ~240 MB)
    excluded_modules = [
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'IPython', 'jupyter',
        'jupyterlab', 'notebook', 'tornado', 'zmq', 'numba', 'llvmlite',
        'pyarrow', 'lxml', 'sqlalchemy', 'h5py', 'cv2', 'skimage',
        'sklearn', 'openpyxl', 'xlsxwriter', 'numexpr', 'bottleneck',
        'seaborn', 'sympy', 'jedi', 'parso',
    ]
    excludes = [f'--exclude-module={m}' for m in excluded_modules]

    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=scipy.special._cdflib',
        '--hidden-import=scipy._lib.array_api_compat.numpy.fft',
        '--hidden-import=scipy.special._ufuncs_cxx',
        '--hidden-import=scipy.linalg.cython_blas',
        '--hidden-import=scipy.linalg.cython_lapack',
        '--hidden-import=scipy.integrate',
        '--hidden-import=scipy.signal',
        '--hidden-import=scipy.ndimage',
        '--hidden-import=scipy.optimize',
        '--hidden-import=matplotlib.backends.backend_tkagg',
    ]

    # Use a local temp directory for ALL PyInstaller I/O — Google Drive's
    # virtual FS does not support the seek/copy operations PyInstaller needs.
    work_root = Path(tempfile.gettempdir()) / "ironlpeaks_build"
    work_root.mkdir(parents=True, exist_ok=True)
    work_path = work_root / "build"
    spec_path = work_root / "spec"
    dist_path = work_root / "dist"
    work_path.mkdir(exist_ok=True)
    spec_path.mkdir(exist_ok=True)
    dist_path.mkdir(exist_ok=True)

    # PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name=IronLPeaks',
        '--onefile',
        '--windowed',
        f'--icon={icon_file}',
        '--clean',
        '--noconfirm',
        f'--workpath={work_path}',
        f'--specpath={spec_path}',
        f'--distpath={dist_path}',
    ]

    if splash_image is not None:
        cmd.append(f'--splash={splash_image}')

    cmd += data_files + hidden_imports + excludes + [str(main_script)]

    print("\nBuilding IronLPeaks...")
    print(f"Work dir: {work_path}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(base_dir))

    if result.returncode == 0:
        temp_exe = dist_path / "IronLPeaks.exe"
        final_dist = base_dir / "dist"
        final_dist.mkdir(exist_ok=True)
        final_exe = final_dist / "IronLPeaks.exe"
        import shutil
        shutil.copy2(str(temp_exe), str(final_exe))
        print(f"\n{'='*60}")
        print(f"Build successful!")
        print(f"Executable: {final_exe}")
        print(f"Size: {final_exe.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"{'='*60}")
    else:
        print(f"\nBuild failed with return code {result.returncode}")


if __name__ == '__main__':
    build()
