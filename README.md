# Predicting 30-Day Hospital Readmission for Patients with Diabetes

A self-learning tutorial that walks through building, evaluating, explaining, and auditing a machine-learning model
that predicts whether a diabetic patient will be readmitted to the hospital within 30 days of discharge.

**Methods:** logistic regression baseline → XGBoost (gradient-boosted trees) → SHAP explainability → subgroup fairness audit
**Data:** [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296), public and de-identified, no credentialing needed

## Quick start

**Google Colab:** upload `readmission_tutorial.ipynb` (or open it from GitHub), uncomment the `!pip install` line in
Step 0, then choose *Runtime → Run all*.

**Local:**
```bash
pip install -r requirements.txt
jupyter lab readmission_tutorial.ipynb     # then Run All
```
The notebook downloads the dataset automatically into `data/` (~3 MB zip). Runtime is about 1–3 minutes on a laptop CPU.
All figures are written to `figures/`, and headline metrics are saved to `figures/results.json`.

## What you will learn

| Step | Topic | Key idea |
|---|---|---|
| 0–1 | Setup & download | Reproducible environment, fixed random seed |
| 2 | Exploratory analysis | Imbalanced outcome (~11%); missing values coded as `?` |
| 3 | Cleaning | Remove deaths/hospice; keep **one encounter per patient** to prevent leakage |
| 4 | Feature engineering | ICD-9 → 9 clinical groups; age bins → numbers; medication summaries |
| 5 | Splitting | Stratified train / validation / test |
| 6 | Baseline | scikit-learn `Pipeline` + logistic regression |
| 7 | XGBoost | Native categorical support, `scale_pos_weight`, early stopping |
| 8 | Evaluation | ROC-AUC vs **PR-AUC**, capacity-based threshold, lift |
| 9 | Explainability | SHAP beeswarm (global) and waterfall (single patient) |
| 10 | Fairness | ROC-AUC by sex, race, and age group |

## Results (test set, 13,998 patients)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.648 | 0.172 |
| XGBoost | **0.658** | **0.187** |
| Random (prevalence) | 0.500 | 0.090 |

Flagging the top 20% highest-risk patients catches **39%** of 30-day readmissions with a PPV of **17.4%**
(**1.94×** lift over random selection). The top predictors are discharge destination, primary diagnosis group, and
number of inpatient visits in the prior year.

## Repository contents

```
readmission_tutorial.ipynb   # the tutorial (code + explanations), run top-to-bottom
requirements.txt             # Python dependencies
figures/                     # generated plots + results.json (created when you run the notebook)
data/                        # downloaded dataset (created automatically; git-ignored)
```

## Exercises

1. Use all encounters with `GroupShuffleSplit` on `patient_nbr` instead of keeping first encounters.
2. Add interaction features (e.g., insulin dose change × A1C result).
3. Tune hyper-parameters with `RandomizedSearchCV` or Optuna.
4. Calibrate probabilities with `CalibratedClassifierCV` and draw a reliability diagram.

## References

- Strack B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 781670.
- Chen T., Guestrin C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Lundberg S., Lee S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
