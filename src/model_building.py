import os
import pickle
import logging

import pandas as pd
import yaml

from sklearn.ensemble import GradientBoostingClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_parameters(file_path: str = "params.yaml") -> dict:
    """
    Load model parameters from YAML file.
    """
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded successfully.")

        return params

    except FileNotFoundError:
        logger.exception(
            "Parameter file not found: %s",
            file_path
        )
        raise

    except yaml.YAMLError:
        logger.exception(
            "Invalid YAML format in: %s",
            file_path
        )
        raise


def load_training_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load feature-engineered training data.
    """
    try:
        logger.info(
            "Loading training data from: %s",
            file_path
        )

        train_data = pd.read_csv(file_path)

        logger.info(
            "Training data loaded successfully. Shape: %s",
            train_data.shape
        )

        return train_data

    except FileNotFoundError:
        logger.exception(
            "Training data file not found: %s",
            file_path
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load training data."
        )
        raise


def prepare_training_data(
    train_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate training features and target variable.
    """
    try:
        X_train = train_data.drop(
            columns=["sentiment"]
        )

        y_train = train_data["sentiment"]

        logger.info(
            "Training features shape: %s",
            X_train.shape
        )

        logger.info(
            "Training target shape: %s",
            y_train.shape
        )

        return X_train, y_train

    except KeyError as error:
        logger.exception(
            "Required target column is missing: %s",
            error
        )
        raise

    except Exception:
        logger.exception(
            "Failed to prepare training data."
        )
        raise


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    learning_rate: float,
    n_estimators: int
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.
    """
    try:
        logger.info(
            "Training Gradient Boosting model "
            "(n_estimators=%d, learning_rate=%s)",
            n_estimators,
            learning_rate
        )

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        logger.info(
            "Model training completed successfully."
        )

        return model

    except Exception:
        logger.exception(
            "Failed during model training."
        )
        raise


def save_model(
    model: GradientBoostingClassifier,
    output_path: str = "models/model.pkl"
) -> None:
    """
    Save trained model to disk.
    """
    try:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        with open(output_path, "wb") as file:
            pickle.dump(model, file)

        logger.info(
            "Model saved successfully to: %s",
            output_path
        )

    except Exception:
        logger.exception(
            "Failed to save model to: %s",
            output_path
        )
        raise


def main() -> None:
    """
    Execute the complete model building pipeline.
    """
    try:
        # Load parameters
        params = load_parameters()

        learning_rate = params[
            "model_building"
        ]["learning_rate"]

        n_estimators = params[
            "model_building"
        ]["n_estimators"]

        # Load training data
        train_data = load_training_data(
            "data/features/train_features.csv"
        )

        # Prepare training data
        X_train, y_train = prepare_training_data(
            train_data
        )

        # Train model
        model = train_model(
            X_train,
            y_train,
            learning_rate,
            n_estimators
        )

        # Save model
        save_model(model)

        logger.info(
            "Model building pipeline "
            "completed successfully!"
        )

    except Exception:
        logger.exception(
            "Model building pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()