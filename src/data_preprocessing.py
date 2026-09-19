import os
import re
import logging

import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Apply NLP preprocessing to a single text sample.
    """

    try:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"http\S+|www\S+|https\S+",
            "",
            text
        )

        # Remove mentions
        text = re.sub(
            r"@\w+",
            "",
            text
        )

        # Remove hashtag symbol
        text = re.sub(
            r"#",
            "",
            text
        )

        # Remove special characters and numbers
        text = re.sub(
            r"[^a-zA-Z\s]",
            "",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    except Exception:
        logger.exception("Failed to preprocess text.")
        raise


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.
    """

    try:
        logger.info("Loading data from: %s", input_path)

        df = pd.read_csv(input_path)

        logger.info(
            "Data loaded successfully. Shape: %s",
            df.shape
        )

        return df

    except FileNotFoundError:
        logger.exception(
            "Input file not found: %s",
            input_path
        )
        raise

    except Exception:
        logger.exception(
            "Failed to load data from: %s",
            input_path
        )
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform NLP preprocessing on the dataset.
    """

    try:
        logger.info(
            "Starting preprocessing. Original shape: %s",
            df.shape
        )

        # Remove rows where content is missing
        df = df.dropna(
            subset=["content"]
        ).copy()

        # Apply NLP preprocessing
        df["content"] = df["content"].apply(
            preprocess_text
        )

        # Remove empty rows after preprocessing
        df = df[
            df["content"].str.strip() != ""
        ].copy()

        logger.info(
            "Preprocessing completed. Final shape: %s",
            df.shape
        )

        return df

    except KeyError as error:
        logger.exception(
            "Required column is missing: %s",
            error
        )
        raise

    except Exception:
        logger.exception(
            "Failed during data preprocessing."
        )
        raise


def save_data(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save processed data to a CSV file.
    """

    try:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Processed data saved to: %s",
            output_path
        )

    except Exception:
        logger.exception(
            "Failed to save processed data to: %s",
            output_path
        )
        raise


def process_file(
    input_path: str,
    output_path: str
) -> None:
    """
    Load, preprocess and save a dataset.
    """

    df = load_data(input_path)

    processed_df = preprocess_data(df)

    save_data(
        processed_df,
        output_path
    )


def main() -> None:
    """
    Execute the complete preprocessing pipeline.
    """

    try:
        process_file(
            "data/raw/train.csv",
            "data/processed/train_processed.csv"
        )

        process_file(
            "data/raw/test.csv",
            "data/processed/test_processed.csv"
        )

        logger.info(
            "Data preprocessing completed successfully!"
        )

    except Exception:
        logger.exception(
            "Data preprocessing pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()