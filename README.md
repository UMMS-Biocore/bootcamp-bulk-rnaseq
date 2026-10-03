# Bootcamp session 4: bulk RNA-seq, why DE uses models

Marimo notebook: `de_models.py`

```bash
# with uv (installs dependencies from the notebook header)
uvx marimo edit --sandbox de_models.py

# or with an existing environment
pip install marimo numpy scipy statsmodels pandas matplotlib
marimo edit de_models.py
```

## Data
* Microarray: GEO GSE8658 (moDCs, HG-U133 Plus 2), downloaded on first run into `data/array/`.
  `data/array/GPL570_probe2symbol.tsv.gz` is the probe → symbol map from GPL570.
* RNA-seq: lab DC raw-count tables, not in this repo. Point `BULK_RNASEQ_DIR` at the folder
  (default `~/Dropbox_umms/projects/carol_chemokine_data`).
