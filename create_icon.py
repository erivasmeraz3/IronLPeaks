"""
Create IronLPeaks icon from the project SVG logo.
Converts the SVG to multi-size ICO and PNG files for the GUI and PyInstaller.
"""

import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
    from PIL import Image

try:
    import cairosvg
    _HAS_CAIROSVG = True
except (ImportError, OSError):
    _HAS_CAIROSVG = False


def svg_to_png_cairosvg(svg_path, png_path, size):
    """Convert SVG to PNG using cairosvg."""
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=size, output_height=size)


def _find_inkscape():
    """Find Inkscape executable."""
    import shutil
    ink = shutil.which('inkscape')
    if ink:
        return ink
    for candidate in [
        r"C:\Program Files\Inkscape\bin\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def svg_to_png_inkscape(svg_path, png_path, size):
    """Convert SVG to PNG using Inkscape CLI."""
    ink = _find_inkscape()
    if ink is None:
        raise FileNotFoundError("Inkscape not found")
    cmd = [
        ink, str(svg_path),
        '--export-type=png',
        f'--export-filename={png_path}',
        f'--export-width={size}',
        f'--export-height={size}',
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def svg_to_png_magick(svg_path, png_path, size):
    """Convert SVG to PNG using ImageMagick."""
    cmd = ['magick', 'convert', '-background', 'none',
           '-resize', f'{size}x{size}', str(svg_path), str(png_path)]
    subprocess.run(cmd, capture_output=True, check=True)


def convert_svg(svg_path, png_path, size):
    """Try available SVG converters in order of preference."""
    if _HAS_CAIROSVG:
        svg_to_png_cairosvg(svg_path, png_path, size)
        return

    for converter in (svg_to_png_inkscape, svg_to_png_magick):
        try:
            converter(svg_path, png_path, size)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    raise RuntimeError(
        "No SVG converter found. Install one of:\n"
        "  pip install cairosvg\n"
        "  Install Inkscape (inkscape.org)\n"
        "  Install ImageMagick (imagemagick.org)"
    )


def create_ico_pillow(images, ico_path):
    """Create ICO file using Pillow."""
    sorted_images = sorted(images, key=lambda x: x.size[0])
    base = sorted_images[-1]
    others = sorted_images[:-1]
    base.save(ico_path, format='ICO', append_images=others, bitmap_format='bmp')


def main():
    base_dir = Path(__file__).parent
    # Look for SVG in common locations
    svg_candidates = [
        Path(r"G:\My Drive\VSCode\Iron L Peaks Logo.svg"),
        base_dir / "Iron L Peaks Logo.svg",
        base_dir.parent / "Iron L Peaks Logo.svg",
    ]

    svg_path = None
    for candidate in svg_candidates:
        if candidate.exists():
            svg_path = candidate
            break

    if svg_path is None:
        print("ERROR: Could not find 'Iron L Peaks Logo.svg'")
        print("Searched:")
        for c in svg_candidates:
            print(f"  {c}")
        sys.exit(1)

    print(f"Using SVG: {svg_path}")

    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []

    print("Creating IronLPeaks icons...")
    for size in sizes:
        png_path = base_dir / f"ironlpeaks_{size}.png"
        print(f"  Generating {size}x{size}...")
        convert_svg(svg_path, png_path, size)
        img = Image.open(png_path).convert('RGBA')
        images.append(img)

    # Save ICO
    ico_path = base_dir / "ironlpeaks.ico"
    print(f"\nCreating ICO: {ico_path}")
    create_ico_pillow(images, ico_path)

    print("Done!")


if __name__ == '__main__':
    main()
