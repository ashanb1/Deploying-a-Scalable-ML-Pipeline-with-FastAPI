# Deploying a Scalable ML Pipeline with FastAPI
Repo: https://github.com/ashanb1/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## Project Overview
This project trains a machine learning classifier on U.S. Census Bureau data to predict whether an individual's income exceeds $50K per year. It includes a full ML pipeline (data preprocessing, training, evaluation, and slice-based performance analysis) and deploys the trained model behind a RESTful API built with FastAPI.

## Environment Setup
This project uses conda. Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate fastapi
```

## Project Structure
- `data/census.csv` — U.S. Census Bureau dataset used for training and evaluation
- `ml/data.py` — data preprocessing functions
- `ml/model.py` — model training, inference, saving/loading, and slice metric functions
- `train_model.py` — full training pipeline script; trains the model and outputs `slice_output.txt`
- `test_ml.py` — unit tests for the ML pipeline
- `main.py` — FastAPI application with GET and POST endpoints
- `local_api.py` — script to test the live API with GET and POST requests
- `model_card.md` — documentation of the trained model, its intended use, and performance
- `model/` — saved model and encoder artifacts (`model.pkl`, `encoder.pkl`)
- `screenshots/` — screenshots showing passing CI, passing unit tests, and a successful local API test

## Running the Pipeline
Train the model and generate slice performance metrics:
```bash
python train_model.py
```

Run the unit tests:
```bash
pytest test_ml.py -v
```

## Running the API
Start the API locally:
```bash
uvicorn main:app --reload
```

In a separate terminal, test it:
```bash
python local_api.py
```

You can also explore the API interactively at `http://127.0.0.1:8000/docs` once it's running.

## Continuous Integration
GitHub Actions runs `flake8` and `pytest` on every push to ensure code quality and test coverage. See `.github/workflows/manual.yml`.