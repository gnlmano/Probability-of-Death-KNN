# Course: Data Science for Decision Making  
## Module: Computational Machine Learning II - Assignment I

### Prediction project: probability of death
- Goal: predict the probability of death of a patient entering an ICU using supervised learning methods, initially K-Nearest Neighbors.
- Dataset: MIMIC-III ICU stays (one row per patient stay). Includes demographics, vitals on ICU admission, lab summaries, main diagnosis (ICD-9), and comorbidities from a separate long-format table.
- Target variable: **HOSPITAL_EXPIRE_FLAG** (binary indicator of in-hospital mortality).
- Constraint: avoid using any feature that would not be available on the first day of ICU stay.
- ICD-9 meaning from MIMIC metadata; additional comorbidities from MIMIC_diagnoses.csv.

---

# Project Summary

### 1. Data Overview
- Each row represents `subject_id + hadm_id + icustay_id`.  
- Columns include age, gender, vitals, labs, and diagnosis codes.  
- Secondary diseases (comorbidities) come from a separate table and must be aggregated manually.  
- The target (mortality) is relatively rare (≈10%), but not extremely imbalanced, so heavy imbalance treatments were not required.

---

### 2. Methodology Pipeline
- Standard preprocessing to reflect what would be known on day 1 of an ICU stay:
  - numerical imputation (median)
  - categorical cleaning and encoding
  - scaling for distance-based models (important for KNN)
  - leakage-free joins of comorbidities and diagnosis tables  
- Stratified train/validation/test split to preserve the mortality proportion.  
- Cross-validation using **StratifiedKFold**.  
- Evaluated metrics: AUC and F2-score (to emphasize recall of mortality cases).

---

### 3. Initial Requirement: Train a KNN Classifier
- Did hyperparameter tuning of KNN (k, weights, metric, distance).  
- Scaling and normalization were applied appropriately.  
- Evaluated performance both with AUC and clinically meaningful F2-score.

---

### 4. Why KNN Fails for This Problem
- Tuning repeatedly collapsed to **k = 1** with **weights='distance'**, meaning KNN essentially memorized the training data.  
- High dimensionality, noisy vitals, and varied ICD-9 patterns made KNN unstable.  
- In-sample performance was artificially perfect; CV looked acceptable; but test performance dropped sharply → clear overfitting.  
- Given the setting (clinical tabular data + thousands of sparse diagnostic codes), KNN is not appropriate for ICU mortality risk modelling.  

**Conclusion:** KNN was used to satisfy assignment requirements, but not selected as the final model.

---

### 5. Moving Beyond KNN: Better Models
- Evaluated:  
  - logistic regression with class weights  
  - random forests  
  - gradient boosted trees (XGBoost)  
- The dataset has only modest imbalance (~10%), and **XGBoost already managed this well** without additional techniques (e.g., SMOTE).  
- XGBoost delivered consistently strong performance on both train and test sets, with stable probability estimates and no signs of overfitting.  
- Therefore, XGBoost became the final deployed model.

---

### 6. Encoding ICD-9 & Diagnoses Without Data Leakage
- ICD-9 codes have high cardinality and encode strong clinical information.  
- Naive target encoding would leak label information across folds.  
- Implemented a **custom leakage-free out-of-fold target encoding**:
  - For each CV fold, compute mortality frequencies only from the training slice.
  - Merge patient-level comorbidities to produce engineered features:
    - **max_mortality**
    - **mean_mortality**
    - **count_comorbidities**
  - Apply global mappings only after the model is finalized.  
- These features turned out to be clinically meaningful and statistically powerful.

---

### 7. Model Explainability (SHAP)
- SHAP analysis confirmed the value of feature engineering:
  - **mean_mortality** and **max_mortality** were the top two predictors.
  - Followed by **age at admission**, **target-encoded primary diagnosis**, and **count of comorbidities**.
  - Next top features came from vitals and lab summaries.  
- Interpretation: engineered diagnosis-based features successfully captured latent disease severity, supporting clinical plausibility.

---

### 8. API Deployment
- Final XGBoost pipeline and ICD-9 encoders bundled using joblib.  
- Built a FastAPI service with two endpoints:
  - `/health` — simple liveness check  
  - `/sampling` — returns a random test-set patient + predicted mortality risk  
- No manual prediction endpoint is provided, because real ICU inputs require structured EHR fields (ICD-9 lists, vitals, device measurements, etc.), not something feasible via manual API entry.  
- Containerized with Docker and ready for deployment to Render, AWS, or GCP.

---

### 9. Limitations & Ethical Considerations
- MIMIC-III represents U.S. hospital data (Beth Israel Deaconess), so generalizability to other countries or healthcare systems is limited.  
- The model is purely academic and not suitable for real clinical decision-making.  
- Differences in EHR structure, coding practices, and ICU workflows would require significant domain adaptation.

---

### 10. Future Work
- Consider calibration strategies (isotonic/Platt) for improved decision thresholds.  
- Add time-series features from continuous vitals rather than static summaries.  
- Explore learned embeddings for ICD-9 codes for richer representations.  
- Validate on external hospital datasets to test generalization.
