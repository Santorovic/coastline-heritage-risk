"""Step 6 - Figures of the article (results/figures/).

Usage
    python scripts/06_figures.py            # every figure
    python scripts/06_figures.py fig8       # one figure (fig2 ... fig9, ext)
    python scripts/06_figures.py list       # list of the figures

The maps (Figures 4, 5, 7 and Extended data Figure 1) are previews: the published
maps were finished in a GIS from the same results.
"""
from coastline.figures.paper import main

if __name__ == "__main__":
    main()
