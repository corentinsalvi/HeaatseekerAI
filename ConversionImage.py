from PIL import Image
from pathlib import Path
import argparse

def convert_folder(folder: Path, recursive: bool = False):
    pattern = "**/*.webp" if recursive else "*.webp"
    count = 0
    for p in folder.glob(pattern):
        try:
            out = p.with_suffix(".jpg")
            if out.exists():
                # si vous voulez écraser, commentez la ligne suivante
                print(f"Skip (exists): {out}")
                continue
            img = Image.open(p).convert("RGB")
            img.save(out, "JPEG")
            count += 1
            print(f"Converted: {p} -> {out}")
        except Exception as e:
            print(f"Error converting {p}: {e}")
    print(f"Done. {count} files converted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .webp images to .jpg")
    # Défaut sur le dossier "images" situé à côté du script :
    parser.add_argument("folder", nargs="?", default=str(Path(__file__).parent / "images"), help="Dossier à traiter")
    parser.add_argument("-r", "--recursive", action="store_true", help="Parcourir les sous-dossiers")
    args = parser.parse_args()

    convert_folder(Path(args.folder), args.recursive)