# Course: Data Science for Decision Making
## Module: Computational Machine Learning II - Assignment I
### K-Nearest-Neighbours

Prediction project: probability of death
In this project, you have to predict the probability of death of a patient that is entering an ICU (Intensive Care Unit), using K-Nearest Neighbor models.

The dataset comes from MIMIC project (https://mimic.physionet.org/). MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

Each row of mimic_train.csv corresponds to one ICU stay (hadm_id+ icustay_id) of one patient (subject_id). Column HOSPITAL_EXPIRE_FLAG is the indicator of death (=1) as a result of the current hospital stay; this is the outcome to predict in our modelling exercise. The remaining columns correspond to vitals of each patient (when entering the ICU), plus some general characteristics (age, gender, etc.), and their explanation can be found at mimic_patient_metadata.csv.

Please don't use any feature that you infer you don't know the first day of a patient in an ICU.

Note that the main cause/disease of patient condition is embedded as a code at ICD9_diagnosis column. The meaning of this code can be found at MIMIC_metadata_diagnose.csv. But this is only the main one; a patient can have co-occurrent diseases (comorbidities). These secondary codes can be found at extra__data/MIMIC_diagnoses.csv.

As performance metric, you can use AUC for the binary classification case, but feel free to report as well any other metric if you can justify that is particularly suitable for this case.

Main tasks are:

1. Using mimic_train.csv file build a predictive model for HOSPITAL_EXPIRE_FLAG*.
2. For this analysis there is an extra test dataset, mimic_test_death.csv. Apply your final model to this extra dataset and produce a prediction csv file in same format as mimic_kaggle_death_sample_submission.csv.
3. As a bonus, try different algorithms for neighbor search and for distance, and justify final selection. Try also different weights to cope with class imbalance and also to balance neighbor proximity. Try to assess somehow confidence interval of predictions.

You can follow those steps in your first implementation:

1. Explore and understand the dataset.
2. Manage missing data.
3. Manage categorical features. E.g. create dummy variables for relevant categorical features, or build an ad hoc distance function.
4. Build a prediction model. Try to improve it using methods to tackle class imbalance.
5. Assess expected accuracy of previous models using cross-validation.
6. Run the predictions on the mimic_test_death.csv test file, and report accuracy by submitting to Kaggle page. Follow same preparation steps (missing data, dummies, etc). Remember that you should be able to yield a prediction for all the rows of the test dataset.
