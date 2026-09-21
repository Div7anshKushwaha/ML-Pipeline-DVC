import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_test_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load TF-IDF feature-engineered test data.
    """
    try:
        logger.info(
            "Loading TF-IDF test data from: %s",
            file_path
        )

        test_data = pd.read_csv(file_path)

        logger.info(
            "TF-IDF test data loaded successfully. Shape: %s",
            test_data.shape
        )

        return test_data

    except FileNotFoundError:
        logger.exception(
            "TF-IDF test data file not found: %s",
            file_path
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load TF-IDF test data."
        )
        raise


def prepare_test_data(
    test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate test features and target variable.
    """
    try:
        X_test = test_data.drop(
            columns=["sentiment"]
        )

        y_test = test_data["sentiment"]

        logger.info(
            "Test TF-IDF features shape: %s",
            X_test.shape
        )

        logger.info(
            "Test target shape: %s",
            y_test.shape
        )

        return X_test, y_test

    except KeyError as error:
        logger.exception(
            "Required target column is missing: %s",
            error
        )
        raise

    except Exception:
        logger.exception(
            "Failed to prepare test data."
        )
        raise


def load_model(
    file_path: str
):
    """
    Load the trained model from disk.
    """
    try:
        logger.info(
            "Loading trained model from: %s",
            file_path
        )

        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info(
            "Model loaded successfully."
        )

        return model

    except FileNotFoundError:
        logger.exception(
            "Model file not found: %s",
            file_path
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load trained model."
        )
        raise


def calculate_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Generate predictions and calculate evaluation metrics.
    """
    try:
        logger.info(
            "Generating predictions on TF-IDF test data."
        )

        # Generate predictions
        y_pred = model.predict(X_test)

        # Probability of positive class
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_pred_proba
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

        logger.info(
            "Evaluation metrics calculated successfully."
        )

        return metrics

    except Exception:
        logger.exception(
            "Failed to calculate evaluation metrics."
        )
        raise


def save_metrics(
    metrics: dict,
    output_path: str = "reports/metrics.json"
) -> None:
    """
    Save evaluation metrics to a JSON file.
    """
    try:
        output_file = Path(output_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_file, "w") as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            "Metrics saved successfully to: %s",
            output_path
        )

    except Exception:
        logger.exception(
            "Failed to save metrics to: %s",
            output_path
        )
        raise


def main() -> None:
    """
    Execute the complete model evaluation pipeline.
    """
    try:
        # Load TF-IDF test data
        test_data = load_test_data(
            "data/features/test_tfidf.csv"
        )

        # Prepare test data
        X_test, y_test = prepare_test_data(
            test_data
        )

        # Load trained model
        model = load_model(
            "models/model.pkl"
        )

        # Calculate metrics
        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        # Save metrics
        save_metrics(metrics)

        logger.info(
            "TF-IDF model evaluation completed successfully!"
        )

        logger.info(
            "Metrics: %s",
            metrics
        )

    except Exception:
        logger.exception(
            "Model evaluation pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()