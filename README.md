# Fake Review Detector

A machine learning project that classifies product reviews as:

- Fake (CG)
- Genuine (OR)

The model is trained using TF-IDF features with Logistic Regression and served through a simple Streamlit web app.

## Project Overview

This repository contains:

- Data preprocessing with NLTK-based text cleaning and stemming
- Model training pipeline using scikit-learn
- Inference pipeline for single-text prediction
- Streamlit interface for interactive testing

## Repository Structure

```text
ML_FakeReviewDetector/
|-- app.py
|-- data/
|   `-- reviews.csv
|-- model/
|   |-- model.pkl
|   `-- tfidf.pkl
|-- src/
|   |-- preprocessing.py
|   |-- train.py
|   `-- predict.py
|-- training.ipynb
|-- requirements.txt
|-- README.md
```

## Dataset

- File: data/reviews.csv
- Main columns used in training:
	- label: CG or OR
	- text: raw review text

Label mapping used by training code:

- CG -> 0 (Fake)
- OR -> 1 (Genuine)

## Prerequisites

- Python 3.9+
- pip (latest recommended)

## 1. Create and Activate a Virtual Environment

Run these commands from the project root.

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note:

- The preprocessing module auto-downloads required NLTK resources (stopwords, punkt) on first run.

## 3. Train the Model

From the repository root:

```bash
python src/train.py
```

Expected result:

- Prints Accuracy and Classification Report in the terminal
- Saves artifacts to:
	- model/model.pkl
	- model/tfidf.pkl

## 4. Test the Model from Terminal (Quick Smoke Test)

After training, run:

```bash
python -c "from src.predict import predict; print(predict('Amazing quality and exactly as described'))"
```

You should get one of:

- Genuine (OR)
- Fake (CG)

Try another sample:

```bash
python -c "from src.predict import predict; print(predict('Perfect size and very comfortable. I love the look and feel.'))"
```

## 5. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501), enter a review, and click Predict.

## End-to-End Run Order

Use this exact sequence on a fresh machine:

1. Create virtual environment
2. Activate virtual environment
3. Install dependencies
4. Train model (creates model files)
5. Run terminal smoke test
6. Launch Streamlit app for interactive testing

## Troubleshooting

- ModuleNotFoundError: Ensure you are running commands from the repository root.
- Missing model files: Run python src/train.py before prediction or app launch.
- NLTK download issues: Ensure internet connectivity on first run.
- Streamlit command not found: Re-activate venv and reinstall dependencies.

## Notes

- training.ipynb is available for notebook-based experimentation.
- The script pipeline in src/ is the recommended path for reproducible training and deployment.

## License

MIT © [champst2004](https://github.com/champst2004/ML_FakeReviewDetector/blob/master/LICENSE)