# Data directories

`data/raw/` contains source CSV files generated explicitly by `scripts/generate_dataset.py` or supplied by a caller. `data/processed/` contains validated, normalized CSV output from the pipeline. Processing never overwrites raw input. Generated CSV files are ignored by Git; the directory placeholders are retained.
