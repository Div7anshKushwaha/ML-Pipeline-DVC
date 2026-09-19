import os
import logging
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_parameters(file_path: str = "params.yaml") -> dict:
    """
    Load parameters from the YAML configuration file.
    """
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded successfully.")
        return params

    except FileNotFoundError:
        logger.exception("Parameter file not found: %s", file_path)
        raise

    except yaml.YAMLError:
        logger.exception("Invalid YAML format in: %s", file_path)
        raise


def load_data(url: str) -> pd.DataFrame:
    """
    Load the dataset from the given URL.
    """
    try:
        logger.info("Loading dataset from URL.")

        df = pd.read_csv(url)

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)

        return df

    except Exception:
        logger.exception("Failed to load dataset.")
        raise


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns, filter sentiments,
    and convert sentiment labels.
    """
    try:
        logger.info("Preparing dataset.")

        # Remove unnecessary column
        df.drop(columns=["tweet_id"], inplace=True)

        # Keep only neutral and sadness
        final_df = df[
            df["sentiment"].isin(["neutral", "sadness"])
        ].copy()

        # Convert sentiment labels
        final_df["sentiment"] = final_df["sentiment"].map({
            "neutral": 1,
            "sadness": 0
        })

        logger.info(
            "Dataset preparation completed. Shape: %s",
            final_df.shape
        )

        return final_df

    except KeyError as error:
        logger.exception(
            "Required column is missing: %s",
            error
        )
        raise

    except Exception:
        logger.exception("Failed during data preparation.")
        raise


def split_data(
    df: pd.DataFrame,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing data.
    """
    try:
        logger.info(
            "Splitting dataset with test_size=%s",
            test_size
        )

        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )

        logger.info(
            "Train shape: %s | Test shape: %s",
            train_data.shape,
            test_data.shape
        )

        return train_data, test_data

    except Exception:
        logger.exception("Failed during train-test split.")
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> None:
    """
    Save training and testing datasets to data/raw.
    """
    try:
        os.makedirs("data/raw", exist_ok=True)

        train_data.to_csv(
            "data/raw/train.csv",
            index=False
        )

        test_data.to_csv(
            "data/raw/test.csv",
            index=False
        )

        logger.info(
            "Training data saved to data/raw/train.csv"
        )

        logger.info(
            "Testing data saved to data/raw/test.csv"
        )

    except Exception:
        logger.exception("Failed to save train/test data.")
        raise


def main() -> None:
    """
    Execute the complete data ingestion pipeline.
    """
    try:
        # Load parameters
        params = load_parameters()

        test_size = params["data_ingestion"]["test_size"]

        # Dataset URL
        data_url = (
            "https://raw.githubusercontent.com/"
            "campusx-official/jupyter-masterclass/"
            "main/tweet_emotions.csv"
        )

        # Load data
        df = load_data(data_url)

        # Prepare data
        final_df = prepare_data(df)

        # Train-test split
        train_data, test_data = split_data(
            final_df,
            test_size
        )

        # Save data
        save_data(train_data, test_data)

        logger.info(
            "Data ingestion completed successfully!"
        )

    except Exception:
        logger.exception(
            "Data ingestion pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()