"""Descarga y cachea el conectoma de FlyWire v783 (fase 2).

Uso:
    python -m data.download_flywire
    python -m data.download_flywire --check   # solo verifica la cache

Fuente: philshiu/Drosophila_brain_model (Shiu et al., 2024, Nature), que publica
la materializacion v783 de FlyWire. Bajamos los dos ficheros sueltos en vez de
clonar el repo: el clon arrastraria ademas la version 630 (~90 MB) que no usamos.

Los ficheros quedan en data/raw/, que esta en .gitignore y es inmutable.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
BASE_URL = "https://media.githubusercontent.com/media/philshiu/Drosophila_brain_model/main"
FALLBACK_URL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"

# Tamanos exactos segun la API de GitHub para el commit 91bdd1e de main.
FILES: dict[str, int] = {
    "Connectivity_783.parquet": 100_804_642,
    "Completeness_783.csv": 3_327_347,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_cached(name: str, expected_size: int) -> bool:
    path = RAW_DIR / name
    return path.is_file() and path.stat().st_size == expected_size


def download(name: str, expected_size: int) -> Path:
    path = RAW_DIR / name
    if is_cached(name, expected_size):
        print(f"{name}: ya en cache ({path.stat().st_size / 1e6:.1f} MB)")
        return path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    last_url_error: Exception | None = None
    for base in (FALLBACK_URL, BASE_URL):
        url = f"{base}/{name}"
        try:
            print(f"{name}: descargando de {base} ...", flush=True)
            with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
                downloaded = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    downloaded += len(chunk)
                    print(f"  {downloaded / 1e6:7.1f} MB", end="\r", flush=True)
            print()
        except Exception as exc:  # red caida, 404, redireccion a LFS...
            last_url_error = exc
            tmp.unlink(missing_ok=True)
            continue
        if tmp.stat().st_size == expected_size:
            tmp.replace(path)
            return path
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        last_url_error = RuntimeError(
            f"tamano inesperado: {got} bytes, se esperaban {expected_size}"
        )
    raise RuntimeError(f"no se pudo descargar {name}: {last_url_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="solo comprobar la cache, no descargar"
    )
    parser.add_argument(
        "--sha", action="store_true", help="imprimir sha256 de cada fichero"
    )
    args = parser.parse_args()

    ok = True
    for name, size in FILES.items():
        if args.check:
            cached = is_cached(name, size)
            print(f"{name}: {'ok' if cached else 'FALTA o tamano incorrecto'}")
            ok &= cached
            continue
        path = download(name, size)
        line = f"{name}: {path.stat().st_size / 1e6:.1f} MB en {path}"
        if args.sha:
            line += f"\n  sha256 {sha256(path)}"
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
