# Urban Facility Supply and Aging Population in Shenzhen

## Overview
This repository contains code and data for analyzing the relationship
between urban facility accessibility and aging population dynamics
in Shenzhen, China (2020-2035).

## Repository Structure
```text
shenzhen_aging/
├── notebooks/
├── src/
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
└── figures/
```

## Setup
### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate shenzhen-aging
```

### Option 2: pip
```bash
pip install -r requirements.txt
```

## Data
- Small data files are included in `data/raw/`
- Full dataset (700MB) is archived on Zenodo: [DOI link]
- To reproduce: download from Zenodo and extract to `data/`

## Notebook Execution Order
Run notebooks in numerical order:
1. `01_data_preparation.ipynb` — Data cleaning and census correction
2. `02_travel_time_analysis.ipynb` — Travel time calculation
3. `03_population_forecast.ipynb` — GPR-based population projection
4. `04_gwr_analysis.ipynb` — GWR spatial regression
5. `05_china_cities.ipynb` — National-level city analysis
6. `06_figures.ipynb` — Generate all figures

## Citation
If you use this code or data, please cite: (your paper reference)

## License
MIT License for code; CC-BY 4.0 for data.
