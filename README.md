# Correcting the SPB Model for Seebeck Coefficient Prediction via Symbolic Regression

M.Tech thesis work (IIT Bhubaneswar, supervisor Dr. Sivaiah Bathula). Builds an experimental thermoelectric dataset from Starrydata2, calibrates the single parabolic band (SPB) model per material family, freezes a leakage-safe paper-grouped evaluation, and learns an interpretable symbolic-regression correction to the SPB baseline. Supporting notebooks audit the available datasets (Starrydata2, JARVIS-DFT, literature-mined databases) and prepare SISSO-ready DFT datasets.

## Contents

| File | What it does |
|---|---|
| `pipeline/01_seebeck_dataset_starrydata.ipynb` | Seebeck dataset from Starrydata2: S, n, T plus other variables |
| `pipeline/02_seebeck_features_preprocessing.ipynb` | Seebeck features: extract, explore, preprocess |
| `pipeline/03_mstar_calibration.ipynb` | Effective-mass (m*) calibration of the SPB baseline |
| `pipeline/04_xgb_ranking_merging.ipynb` | XGBoost: input ranking and data-driven family merging |
| `pipeline/05_dimensionless_xgb.ipynb` | Dimensionless variables + XGBoost rerun |
| `pipeline/06_freeze_setup.ipynb` | Freeze the setup: locked test papers, final inputs, frozen config, baseline ladder |
| `pipeline/07_pysr_residual.ipynb` | PySR residual equation on top of the family-calibrated SPB baseline |
| `pipeline/Seebeck_SPB_pipeline.ipynb` | Seebeck coefficient from transport data: SPB baseline, family calibration and correction models |
| `data_audits/seebeck_audit_colab_1.ipynb` | Seebeck dataset audit: what comes with Seebeck, and how much is usable? |
| `data_audits/starrydata_te_audit.ipynb` | Starrydata thermoelectric audit: material families, Seebeck–carrier matching, n/p coverage |
| `data_audits/starrydata2_family_audit_colab_1.ipynb` | Starrydata2 family / process / variable audit (Colab version) |
| `data_audits/SPB_Data_Feasibility_Pooled_and_Family.ipynb` | Is there enough data for pooled and single-family SPB residual models? |
| `data_audits/thermoelectric_data_feasibility_colab_1.ipynb` | Thermoelectric data feasibility check (DFT + literature-mined experimental data) |
| `data_audits/thermoelectric_residual_availability_check.ipynb` | Is there enough data to compute the SPB residual? A direct check |
| `data_audits/thermoelectric_datasets_exploration.ipynb` | Thermoelectric Seebeck datasets — first look |
| `data_audits/thermoelectric_seebeck_datasets_study.ipynb` | Thermoelectric Seebeck Datasets: Exploratory Study and SPB Benchmark |
| `data_audits/SISSO_seebeck_check.ipynb` | Checking the SISSO equations of Yuan et al., *Mater. Today Phys.* 25 (2022) 100706 |
| `data_audits/chemdataextractor_carrier_concentration_check.ipynb` | Is carrier concentration hiding in ChemDataExtractor after all? |
| `data_audits/find_source_paper_dois.ipynb` | Extract the exact DOI list behind the DFT/experiment overlap set |
| `data_audits/TE_Seebeck_Family_Counter.ipynb` | Seebeck coefficient counts by material family, across DFT-only databases |
| `dft_sisso/JARVIS_TE_Families.ipynb` | JARVIS-DFT Thermoelectric Data Extraction |
| `dft_sisso/JARVIS_TE_SISSO_Final.ipynb` | JARVIS-DFT Thermoelectric Data -> SISSO-Ready Dataset |
| `dft_sisso/HalfHeusler_SISSO_Ready.ipynb` | JARVIS-DFT Half-Heusler Thermoelectric Data -> SISSO-Ready Dataset |
| `dft_sisso/HalfHeusler_XGBoost_Symbolic.ipynb` | Half-Heusler Seebeck: XGBoost Feature Importance + Symbolic Regression |
| `dft_sisso/Skutterudite_SISSO_Ready.ipynb` | JARVIS-DFT Skutterudite Thermoelectric Data -> SISSO-Ready Dataset |
| `earlier_pipeline/thesis_pipeline_full_colab.ipynb` | Thermoelectric residual-learning pipeline — full, run-it-yourself version |
| `earlier_pipeline/thesis_pipeline_steps1to6.ipynb` | Thermoelectric residual-learning pipeline — steps 1-6, code and results |
| `scripts/build_tiers.py` | build tiers (script) |
| `scripts/jarvis_te_families.py` | jarvis te families (script) |
| `scripts/fetch_dft_and_literature_mined_db.py` | fetch dft and literature mined db (script) |

Notebooks are saved with the outputs of their last run, so results and figures can be read on GitHub without running anything.

## Data

Experimental data come from the Starrydata2 open thermoelectric database; DFT data from JARVIS-DFT (`jarvis-tools`). Data files are not committed; the notebooks download or rebuild them.

## How to run

The notebooks were developed on Google Colab. Upload a notebook to Colab (File → Upload notebook) and run the cells in order; the first cells install anything missing. To run locally in VS Code instead: `python -m venv .venv`, activate it, `pip install -r requirements.txt jupyter`, then open the notebook and pick the `.venv` kernel. Cells that mount Google Drive (`google.colab`) can be skipped locally if the data files are placed next to the notebook.

Python packages used: `gplearn`, `ipython`, `jarvis-tools`, `matminer`, `matplotlib`, `mp-api`, `numpy`, `openpyxl`, `pandas`, `pymatgen`, `pysr`, `requests`, `scikit-learn`, `scipy`, `seaborn`, `shap`, `sympy`, `tqdm`, `urllib3`, `xgboost` (see `requirements.txt`).

## License

MIT, see `LICENSE`.
