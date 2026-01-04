# Repository-specific instructions for AI coding agents

Purpose: Give concise, actionable guidance to quickly make safe, useful edits in this repo.

- **Big picture:** This is a learning / data-engineering ML repo that processes CSV emission datasets and uses notebooks for exploration. Data flows: `data/raw/<sector>/` (source CSVs) -> processing notebooks/scripts -> `data/processed/` (derived CSVs).

- **Key locations:**
  - `data/raw/` : sector folders (e.g., `agriculture`, `forestry_and_land_use`) containing source CSVs such as `*_country_emissions.csv` and `*_emissions-sources.csv`.
  - `data/processed/` : processed outputs. Do not overwrite raw files; write derived artifacts here.
  - `notebooks/` : exploratory and ETL notebooks. Prefer small, targeted edits here and preserve notebook cell structure.
  - `requirements.txt` : canonical Python dependencies used during development (see install command below).

- **Environment & run commands (Windows / dev):**
  - Activate the project's environment (shown in terminal prompt): `conda activate greenHouse_Emission_predictor` or use your venv activation method.
  - Install deps: `pip install -r requirements.txt` (repo has been validated with these packages).
  - Launch notebooks: `jupyter notebook` or `jupyter lab`.

- **Patterns & conventions discovered in repo:**
  - CSV naming: source files follow `topic_country_emissions.csv` and `topic_emissions-sources.csv` patterns across `data/raw/*`.
  - Notebooks import stack commonly used: `pandas`, `numpy`, `geopandas`, `shapely`, `reverse_geocoder`, `matplotlib`, `seaborn`. Check for typos in notebook imports (example: some notebooks mistakenly import `geopanas` instead of `geopandas`).
  - Notebooks are primarily for data engineering; prefer adding small reusable helper functions or separate `.py` utilities if logic grows.

- **Safety & repo hygiene rules for AI edits:**
  - Do not modify files in `data/raw/` (source data). If preprocessing is needed, write outputs to `data/processed/` and document transformation in `data/processed/README.md`.
  - When editing notebooks, keep changes minimal and preserve markdown context. If converting notebook logic to scripts, place them in a new `src/` folder and add an entry to `README.md` describing the script.
  - Avoid broad refactors unless requested. The codebase is primarily educational — prefer explicit, well-commented changes.

- **Examples of actionable edits:**
  - Fix import typos in notebooks (e.g., change `import geopanas as gpd` -> `import geopandas as gpd`).
  - Add a small helper `scripts/` Python file to centralize repeated CSV parsing and write results to `data/processed/`.
  - Add a README note documenting a new processing step in `data/processed/README.md`.

- **Cross-component notes & integration points:**
  - Downstream work assumes processed CSVs in `data/processed/` with consistent column names. Preserve column names or add schema-migration steps.
  - Geoprocessing utilities rely on `geopandas` and `shapely`; reverse geocoding uses `reverse_geocoder`. Keep those dependencies if adding geo features.

- **What I couldn't find (ask before changing):**
  - No centralized `src/` package or tests found — confirm preferred location for new Python modules.
  - No CI/test commands discovered; ask how the user prefers running checks before submitting larger PRs.

If anything above is unclear or you want the file tuned for stricter rules (commit hooks, linters, CI steps), tell me which areas to expand.
