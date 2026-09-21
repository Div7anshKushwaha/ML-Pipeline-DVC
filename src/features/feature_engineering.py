import os
import logging

import pandas as pd
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_parameters(file_path: str = "params.yaml") -> dict:
    """
    Load feature engineering parameters from YAML file.
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


def load_data(
    train_path: str,
    test_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed training and testing data.
    """
    try:
        logger.info("Loading processed training data.")
        train_data = pd.read_csv(train_path)

        logger.info("Loading processed testing data.")
        test_data = pd.read_csv(test_path)

        logger.info(
            "Train shape: %s | Test shape: %s",
            train_data.shape,
            test_data.shape
        )

        return train_data, test_data

    except FileNotFoundError:
        logger.exception(
            "Processed data file not found."
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load processed data."
        )
        raise


def create_tfidf_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create TF-IDF features using TfidfVectorizer.
    """

    try:
        logger.info(
            "Creating TF-IDF features with max_features=%d",
            max_features
        )

        # Separate features and target
        X_train = train_data["content"]
        y_train = train_data["sentiment"]

        X_test = test_data["content"]
        y_test = test_data["sentiment"]

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features
        )

        # Fit ONLY on training data
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Transform test data using training vocabulary
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info(
            "TF-IDF transformation completed."
        )

        logger.info(
            "Vocabulary size: %d",
            len(vectorizer.get_feature_names_out())
        )

        # Convert sparse matrices to DataFrames
        feature_names = vectorizer.get_feature_names_out()

        X_train_df = pd.DataFrame(
            X_train_tfidf.toarray(),
            columns=feature_names
        )

        X_test_df = pd.DataFrame(
            X_test_tfidf.toarray(),
            columns=feature_names
        )

        # Add target column
        X_train_df["sentiment"] = y_train.values
        X_test_df["sentiment"] = y_test.values

        logger.info(
            "Feature engineering completed. "
            "Train shape: %s | Test shape: %s",
            X_train_df.shape,
            X_test_df.shape
        )

        return X_train_df, X_test_df

    except KeyError as error:
        logger.exception(
            "Required column missing: %s",
            error
        )
        raise

    except Exception:
        logger.exception(
            "Failed during TF-IDF feature engineering."
        )
        raise


def save_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Save TF-IDF feature-engineered training and testing data.
    """

    try:
        os.makedirs(
            "data/features",
            exist_ok=True
        )

        train_features.to_csv(
            "data/features/train_tfidf.csv",
            index=False
        )

        test_features.to_csv(
            "data/features/test_tfidf.csv",
            index=False
        )

        logger.info(
            "Training TF-IDF features saved to "
            "data/features/train_tfidf.csv"
        )

        logger.info(
            "Testing TF-IDF features saved to "
            "data/features/test_tfidf.csv"
        )

    except Exception:
        logger.exception(
            "Failed to save TF-IDF feature-engineered data."
        )
        raise


def main() -> None:
    """
    Execute the complete feature engineering pipeline.
    """

    try:
        # Load parameters
        params = load_parameters()

        max_features = params[
            "feature_engineering"
        ]["max_features"]

        # Input paths
        train_path = (
            "data/processed/train_processed.csv"
        )

        test_path = (
            "data/processed/test_processed.csv"
        )

        # Load data
        train_data, test_data = load_data(
            train_path,
            test_path
        )

        # Create TF-IDF features
        train_features, test_features = create_tfidf_features(
            train_data,
            test_data,
            max_features
        )

        # Save features
        save_features(
            train_features,
            test_features
        )

        logger.info(
            "TF-IDF feature engineering pipeline "
            "completed successfully!"
        )

    except Exception:
        logger.exception(
            "Feature engineering pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()