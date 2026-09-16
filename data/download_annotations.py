"""Descarga las anotaciones publicas de FlyWire v783 con la posicion de cada neurona.

Uso:
    python -m data.download_annotations

Fuente: flyconnectome/flywire_annotations (Schlegel et al., 2024, Nature), fichero
suplementario 1. No hace falta registro en FlyWire Codex. Trae por `root_id` la
posicion de cada neurona (`pos_x/y/z`, en voxeles de 4 x 4 x 40 nm), su clase y su
hemisferio. El repo de Shiu no trae coordenadas; las usa solo el visor anatomico.

Los `root_id` se cruzan con los indices del conectoma por la fila de
Completeness_783.csv: el indice i de Shiu es la fila i de ese CSV (verificado sobre
las 138.639 neuronas del parquet).
"""

from __future__ import annotations

import sys

from data import download_flywire as flywire

URL = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files"
NAME = "Supplemental_file1_neuron_annotations.tsv"
SIZE = 31_718_505  # segun la API de GitHub, rama main, 16-09-2026


def main() -> int:
    # Reutiliza el descargador del conectoma (cache, .part, tamano exacto).
    flywire.FALLBACK_URL = URL
    flywire.BASE_URL = URL
    path = flywire.download(NAME, SIZE)
    print(f"{NAME}: {path.stat().st_size / 1e6:.1f} MB en {path}")
    print(f"  sha256 {flywire.sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
