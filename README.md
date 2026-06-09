# Credit Risk Pipeline

This project is a modular credit risk analysis pipeline built in Python. It takes a credit risk dataset, cleans and preprocesses the data, engineers additional features, trains a model, generates predictions, and saves outputs. The structure is split into separate Python files so each part of the workflow is easier to manage, test, and extend.

## Project Structure

### `main.py`
The main entry point for the project. This file runs the full pipeline in order:
- loads the dataset
- renames columns into a standard format
- cleans and preprocesses the data
- engineers new features
- trains the model
- generates predictions
- saves outputs

### `src/config.py`
Stores configuration settings such as file paths, model settings, constants, or reusable parameters.

### `src/loader.py`
Handles loading the raw CSV dataset into a pandas DataFrame.

### `src/schema_inference.py`
Standardizes the dataset column names by mapping original variable names into cleaner canonical names used throughout the pipeline.

### `src/preprocessing.py`
Performs data cleaning and preprocessing steps such as:
- handling missing values
- encoding categorical variables
- preparing the data for modeling

### `src/feature_engineering.py`
Creates additional derived features from the cleaned data to improve model performance and analysis.

### `src/eda.py`
Contains exploratory data analysis code, such as summary statistics and visualizations used to better understand the dataset.

### `src/train.py`
Handles splitting the data into training and testing sets and training the machine learning model.

### `src/evaluate.py`
Evaluates model performance using metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

### `src/predict.py`
Generates predictions from the trained model, including predicted labels, probabilities, and risk levels.

### `src/segmentation.py`
Contains customer segmentation logic, such as clustering borrowers into groups based on financial characteristics.

### `src/report_generator.py`
Responsible for generating a final report summarizing results, predictions, and insights. This can later be extended into PDF report generation.

### `src/util.py`
Stores helper functions used across multiple files.

## Data Folder

### `data/`
Contains the input dataset used for the project.

## Outputs Folder

### `outputs/`
Stores generated outputs such as:
- processed datasets
- predictions
- evaluation results
- future reports or plots

## Setup

Install the required packages with:

```bash
pip install -r requirements.txt