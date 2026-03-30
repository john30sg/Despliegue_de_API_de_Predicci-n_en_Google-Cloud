
from fastapi import FastAPI
import pandas as pd
from joblib import load
from datetime import datetime
import pytz
from google.cloud import bigquery
import os



app = FastAPI()



@app.post("/predict")
async def predict_houseprice():
    # Load environment variables
    GCP_PROJECT = os.getenv("GCP_PROJECT")
    GCP_DATASET = os.getenv("GCP_DATASET")
    GCP_FEATURES_TABLE = os.getenv("GCP_FEATURES_TABLE")
    GCP_PREDICTIONS_TABLE = os.getenv("GCP_PREDICTIONS_TABLE")

    if not all([GCP_PROJECT, GCP_DATASET, GCP_FEATURES_TABLE, GCP_PREDICTIONS_TABLE]):
        return {"error": "Missing one or more required environment variables: GCP_PROJECT, GCP_DATASET, GCP_FEATURES_TABLE, GCP_PREDICTIONS_TABLE"}

    classifier = load("linear_regression.joblib")
    client = bigquery.Client()

    features_query = f"""
        SELECT string_field_0 as feature_name
        FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_FEATURES_TABLE}`
    """
    query_job = client.query(features_query)
    feature_rows = query_job.result()
    features = [row.feature_name for row in feature_rows]

    prediction_query = f"""
        SELECT *
        FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_PREDICTIONS_TABLE}`
    """
    df = client.query(prediction_query).to_dataframe()
    df = df[features]
    predictions = classifier.predict(df)

    now = datetime.now()
    predictions_df = pd.DataFrame({
        'file_name': 'test',
        'prediction': predictions,
        'created_at': now
    })

    predictions_df.to_gbq(
        destination_table=f'{GCP_DATASET}.{GCP_PREDICTIONS_TABLE}',
        project_id=GCP_PROJECT,
        if_exists='append'
    )

    return {
        "predictions": predictions.tolist()
    }