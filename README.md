# Job Change Prediction

This project predicts whether a candidate is likely to look for a new job, using information about their education, experience, current employer and training.

The goal was to build and evaluate a classification model, then make it available through a simple API.

## Dataset

The project uses the HR Analytics: Job Change of Data Scientists dataset. It contains candidate and employment-related information, with the target indicating whether a person is looking for a job change.

## Modelling

I explored the dataset, prepared the features and compared three models:

| Model | Validation AUC |
|---|---:|
| Decision Tree | 0.811 |
| Random Forest | 0.816 |
| XGBoost | 0.824 |

XGBoost performed best, so I tuned its parameters and used it as the final model. The final validation AUC was **0.8236**.

The final model was trained on the combined training and validation data. The fitted model and DictVectorizer are saved together in `model.bin`.

## Running the API

The project uses FastAPI for predictions and Docker for deployment.

To run it locally:

```bash
uv sync
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to try the prediction endpoint.

The API accepts candidate details through `POST /predict` and returns a probability of looking for a job change, along with a prediction based on a 0.5 threshold.

## Running with Docker

Build the image:

```bash
docker build -t job-change-prediction .
```

Run the container:

```bash
docker run -p 8000:8000 job-change-prediction
```

Then open `http://localhost:8000/docs`.

## Project Structure

- `notebooks/` – Exploratory data analysis and modelling
- `data/` – Raw and processed datasets
- `predict.py` – FastAPI application and prediction logic
- `model.bin` – Saved model and DictVectorizer
- `Dockerfile` – Docker configuration

## Technologies

Python, pandas, scikit-learn, XGBoost, FastAPI, uv, and Docker.