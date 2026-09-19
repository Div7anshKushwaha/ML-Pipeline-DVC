# ML Pipeline with DVC

An end-to-end, reproducible Natural Language Processing (NLP) classification pipeline built with Python and [DVC](https://dvc.org/). The project demonstrates how to move from a notebook-based experiment toward a modular and maintainable machine learning workflow.

> **Project goal:** The objective of this project is not to build a perfect model. It is to understand how an industry-style machine learning pipeline is organized, versioned, reproduced, and improved over time.

## Overview

This project performs binary sentiment classification on tweets. It classifies tweets into two categories:

- `neutral` → `1`

- `sadness` → `0`

The workflow is divided into five DVC stages:

```
Raw dataset
    ↓
Data ingestion
    ↓
Text preprocessing
    ↓
Feature engineering with Bag of Words
    ↓
Gradient Boosting model training
    ↓
Model evaluation
    ↓
reports/metrics.json
```

The complete pipeline can be reproduced with one command:

```bash
dvc repro
```

DVC uses the pipeline definition, dependencies, parameters, and lock file to determine which stages need to run.

## Why DVC?

A machine learning project includes more than source code. It also depends on datasets, generated features, trained models, parameters, and evaluation results. DVC helps connect these dependencies and makes experiments reproducible.

This project uses DVC to:

- Define the pipeline in `dvc.yaml`.

- Lock reproducible stage states in `dvc.lock`.

- Track configurable parameters through `params.yaml`.

- Store evaluation results in `reports/metrics.json`.

- Compare results across Git revisions with `dvc metrics diff`.

- Visualize stage dependencies with `dvc dag`.

A DVC remote is not configured yet. Therefore, the current workflow reproduces the pipeline from the files available in the repository. A remote can be added later for sharing datasets and model artifacts across environments.

## Pipeline stages

### 1. Data ingestion

The ingestion stage loads the raw train and test datasets and prepares the data for subsequent stages.

### 2. Text preprocessing

The preprocessing stage cleans tweet text by:

- Converting text to lowercase.

- Removing URLs and user mentions.

- Removing hashtag symbols.

- Removing non-alphabetic characters.

- Normalizing whitespace.

- Removing empty records.

For example:

```
Before:  Check this amazing product! https://example.com @user #awesome
After:   check this amazing product awesome
```

### 3. Feature engineering

The project converts the cleaned text into numerical features using Scikit-learn's `CountVectorizer`. The vectorizer is fitted only on the training data and then applied to the test data. This prevents the test set from influencing the training vocabulary.

The maximum number of features is controlled through `params.yaml`.

### 4. Model training

The current classifier is a Scikit-learn `GradientBoostingClassifier` with configurable training parameters.

The trained model is serialized as a pickle artifact at:

```
models/model.pkl
```

### 5. Model evaluation

The evaluation stage calculates the following metrics:

- Accuracy

- Precision

- Recall

- ROC-AUC

The results are written to:

```
reports/metrics.json
```

## Current metrics

The following values are a snapshot of the current experiment. They may change when the parameters or data are updated.

| Metric | Value |
| --- | --- |
| Accuracy | 0.6578 |
| Precision | 0.6697 |
| Recall | 0.9031 |
| ROC-AUC | 0.6549 |

The purpose of these metrics is to demonstrate experiment tracking and comparison. They are not intended to represent a final production model.

## Project structure

```
ML-Pipeline-DVC/
├── .dvc/
│   ├── .gitignore
│   └── config
├── docs/
│   ├── Makefile
│   ├── commands.rst
│   ├── conf.py
│   ├── getting-started.rst
│   ├── index.rst
│   └── make.bat
├── notebooks/
│   └── .gitkeep
├── references/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   ├── .gitkeep
│   └── metrics.json
├── src/
│   ├── data/
│   │   ├── .gitkeep
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/
│   │   ├── .gitkeep
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── model_building.py
│   │   └── model_evaluation.py
│   └── visualization/
│       └── .gitkeep
├── .dvcignore
├── .gitignore
├── dvc.lock
├── dvc.yaml
├── LICENSE
├── Makefile
├── params.yaml
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

The repository follows a structured data-science project layout. Data-processing code is organized under `src/data`, feature engineering under `src/features`, and model training and evaluation under `src/models`. The `reports` directory stores evaluation outputs, while `docs`, `notebooks`, `references`, and `src/visualization` provide locations for future documentation, analysis, references, and visualizations.

The raw datasets and generated model artifacts are managed through the DVC workflow and are not represented as committed files in the GitHub tree. The tracked evaluation file is located at `reports/metrics.json`.

## Configuration

Pipeline and model parameters are stored in `params.yaml`, so experiments can be run without changing the Python source code.

The current configuration includes:

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 50

model_building:
  learning_rate: 0.1
  n_estimators: 100
```

For example, the model parameters can be changed as follows:

```yaml
model_building:
  learning_rate: 0.05
  n_estimators: 200
```

Run the pipeline again after changing the parameters:

```bash
dvc repro
```

DVC will rerun only the stages affected by the parameter change.

## Getting started

### Prerequisites

Install the following tools before starting:

- Python 3.9 or newer

- Git

- DVC

### Clone the repository

```bash
git clone https://github.com/Div7anshKushwaha/ML-Pipeline-DVC.git
cd ML-Pipeline-DVC
```

### Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\\Scripts\\activate
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Install DVC if it is not already available:

```bash
pip install dvc
```

### Reproduce the pipeline

```bash
dvc repro
```

### Inspect the results

Display the current metrics:

```bash
dvc metrics show
```

Visualize the pipeline DAG:

```bash
dvc dag
```

Check the pipeline status:

```bash
dvc status
```

Compare metrics between Git revisions:

```bash
dvc metrics diff
```

## Useful DVC commands

| Command | Purpose |
| --- | --- |
| `dvc repro` | Reproduce the pipeline and rerun changed stages |
| `dvc dag` | Display the pipeline dependency graph |
| `dvc metrics show` | Display tracked metrics |
| `dvc metrics diff` | Compare metrics across revisions |
| `dvc status` | Check whether the pipeline is up to date |
| `dvc push` | Upload DVC-tracked artifacts after configuring a remote |
| `dvc pull` | Download artifacts from a configured remote |

## Reproducibility workflow

A typical experiment follows these steps:

1. Change a parameter in `params.yaml`.

1. Run `dvc repro`.

1. Review the updated values in `reports/metrics.json`.

1. Compare the results with `dvc metrics diff`.

1. Commit the source code, parameter changes, DVC metadata, and metrics to Git.

1. Use the Git history to reproduce or compare previous experiments.

This workflow separates experimentation from manual file management and makes the impact of parameter changes easier to inspect.

## Technology stack

| Technology | Role |
| --- | --- |
| Python | Pipeline implementation |
| Pandas | Data loading and manipulation |
| Scikit-learn | Feature engineering, model training, and evaluation |
| PyYAML | Parameter configuration |
| DVC | Data, artifact, and pipeline versioning |
| Git | Source-code version control |
| GitHub | Repository hosting |

## Limitations and next steps

This repository is the first step in an ongoing MLOps learning project. The current model and pipeline are intentionally simple so that the focus remains on understanding reproducible workflow design.

Planned improvements include:

- Configure a DVC remote for shared data and model artifacts.

- Add MLflow for experiment tracking.

- Add automated unit and integration tests.

- Add data and model validation.

- Dockerize the training and serving environments.

- Add GitHub Actions for continuous integration and continuous delivery.

- Build a model-serving API.

- Deploy the service to the cloud.

- Add production monitoring and drift detection.

- Compare additional models and feature representations.

## Learning objective

The central lesson from this project is that machine learning engineering is not only about training a model. It is also about building systems that other people can reproduce, inspect, maintain, and improve.

This repository represents a transition from a standalone notebook toward a versioned ML workflow. The project will continue to evolve as new MLOps tools and production practices are added.

## Author

**Divyansh Kushwaha**
BS in Data Science and Applications, IIT Madras

GitHub: [Div7anshKushwaha](https://github.com/Div7anshKushwaha)

## References

[1]: https://dvc.org/doc "DVC Documentation"

[2]: https://scikit-learn.org/stable/ "Scikit-learn Documentation"

[3]: https://docs.python.org/3/ "Python Documentation"

[4]: https://git-scm.com/doc "Git Documentation"

[5]: https://pandas.pydata.org/docs/ "Pandas Documentation"

[6]: https://github.com/Div7anshKushwaha/ML-Pipeline-DVC "ML-Pipeline-DVC Repository"

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.

## Repository

[View the project on GitHub](https://github.com/Div7anshKushwaha/ML-Pipeline-DVC)