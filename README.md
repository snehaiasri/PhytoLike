# PhytoLike

**PhytoLike** is an AI-enabled web platform for predicting **phytochemical-likeness** from molecular structure.  
It estimates whether a query small molecule lies closer to **plant-derived phytochemical space** than to a non-plant background using SMILES-based input.

## Features

- Single-compound prediction from **SMILES**
- Batch prediction through **CSV upload**
- Continuous **phytochemical-likeness score**
- Interpretable prediction category
- Similarity-based confidence estimate
- Nearest training-neighbor display

## Project structure

- `app_phytolike_final_ui_v5.py` — Streamlit web application
- `train_phytochemical_likeness_phase1_v2.py` — model training script
- `predict_phytochemical_likeness.py` — standalone prediction script
- `seed_phytochemical_likeness_with_smiles.csv` — pilot dataset
- `phytolike_home_illustration.svg` — homepage illustration

## Installation

Create and activate your environment, then install the required packages:

```bash
pip install -r requirements_phytolike_phase1.txt
```

## Run the app

```bash
streamlit run app_phytolike_final_ui_v5.py
```

## Model development

Train the pilot model using:

```bash
python train_phytochemical_likeness_phase1_v2.py
```

This will generate the trained model file required by the Streamlit app.

## Input format

### Single prediction
Provide a valid **SMILES** string in the app interface.

### Batch prediction
Upload a CSV file containing at least:

- `smiles`

Optional columns:
- `compound_id`
- `compound_name`

## Output

For each query compound, PhytoLike returns:

- canonical SMILES
- phytochemical-likeness score
- prediction label
- nearest-neighbor similarity
- confidence estimate

## Current scope

This repository contains a **pilot prototype** developed for proof-of-concept demonstration and abstract submission.  
The full version will be expanded using a larger curated dataset and more comprehensive model development.

## Contact

**Sneha Murmu, Ph.D.**  
Scientist (Bioinformatics)  
Division of Agricultural Bioinformatics  
ICAR-Indian Agricultural Statistics Research Institute, New Delhi  
Email: murmu.sneha07@gmail.com
