#!/usr/bin/env python
"""Download raw geo layers for one city into data/raw/geo/<city>/.

Usage: python scripts/fetch_geo_durham.py   (or fetch_geo_ramat_gan.py — both
just call fetch_geo.main with their own city name; edit that module, not these).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fetch_geo import main

if __name__ == "__main__":
    main("durham")
