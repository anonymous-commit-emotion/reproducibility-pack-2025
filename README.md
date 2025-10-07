# Reproducibility Pack for "Detecting Developer Emotions and Its Correlation with Project Health"

This repository contains the code, data, and instructions to reproduce the key findings of our paper, "Detecting Developer Emotions in GitHub Commit Messages and Its Correlation with Project Health."

Our work introduces **CommiTune**, a fine-tuned CodeBERT model that achieves state-of-the-art performance in detecting emotions from commit messages, and provides a rigorous analysis of the long-term relationship between developer emotions and project health metrics.

## Requirements

To run the experiments, you will need Python 3.9+ and the libraries listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

**Key libraries include**: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `huggingface_hub`, `statsmodels`, and `matplotlib`.

## Dataset

The datasets used in this study are located in the `/data` directory:
* `manual_labeled_2k.csv`: The 2,000-commit gold-standard dataset with four emotion labels (`Satisfaction`, `Frustration`, `Caution`, `Neutral`) used for training and testing our models.
* `commits_with_emotions_final_20k.csv`: The full 20,000-commit dataset annotated with emotions by our final CommiTune model.
* `project_health_master_1.6k.csv`: The time-series dataset with project health metrics aggregated quarterly for each repository.

## How to Reproduce Our Results

Follow these steps to reproduce our main findings.

### Step 1: Data Cleaning and Project Health Metric Construction

The initial data cleaning and aggregation of project health metrics are performed by the following scripts. Pre-computed output data is already provided, but you can re-run these to verify the process.

1.  **Text Cleaning**: The `cleaning_from_4M.ipynb` notebook details the extensive regex-based cleaning applied to the raw commit messages.
2.  **Health Metric Calculation**: The `build_project_health_master.py` script computes bug-fix rates, productivity metrics, and code churn, aggregating them into the quarterly project health dataset.

### Step 2: Reproduce the CommiTune Model (Macro F1 = 0.88)

Our state-of-the-art model, **CommiTune**, was created by fine-tuning CodeBERT on a synthetically augmented dataset.

* To reproduce the final evaluation, open the `fine-tuning-emollm-evaluation.ipynb` notebook.
* Load the final CommiTune model from the Hugging Face Hub (see link below).
* Run the evaluation cells against the 400-sample held-out test set from `manual_labeled_2k.csv`. This will replicate the **Macro F1 of 0.88** and **Accuracy of 86%** reported in the paper.

The data augmentation process is detailed in `augmentation-llama.ipynb`.

### Step 3: Reproduce the Correlation Analysis

The statistical analysis exploring the link between long-term emotion and project health is in the `correlation-analysis.ipynb` notebook.

* Run the notebook cells in the **"Long-Term (Repository-Level) Analysis"** section.
* This will generate the correlation heatmaps (Figures 9.2 and 9.3 in the paper) and the statistical results demonstrating that long-term aggregated emotions are strong indicators of project culture and risk.

## Citation

If you use this code or data in your research, please cite our paper:

```
[Anonymous, 2025]
```
