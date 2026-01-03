

# End-to-End Data Science Project

## Overview
This project demonstrates a complete **end-to-end machine learning pipeline**, covering data ingestion to model evaluation using best practices followed in real-world data science projects.

The pipeline is modular, configurable, and production-ready, enabling reproducibility and scalability.

## ML Pipeline Workflow
1. **Data Ingestion**
   - Load and store raw data from source
2. **Data Validation**
   - Validate dataset structure using schema.yaml
3. **Data Transformation**
   - Feature engineering
   - Data preprocessing
4. **Model Training**
   - Train machine learning models using configurable parameters
5. **Model Evaluation**
   - Track experiments and metrics using **MLflow** and **DagsHub**

## Project Structure
- `config.yaml` – Configuration for pipeline execution
- `schema.yaml` – Dataset validation rules
- `params.yaml` – Model hyperparameters
- `src/config` – Configuration manager
- `src/components` – Modular pipeline components
- `src/pipeline` – Training and evaluation pipeline
- `main.py` – Entry point to run the full pipeline

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- MLflow
- DagsHub
- YAML configuration
- Modular project architecture

## How to Run
```bash
python main.py
