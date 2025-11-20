## Why KNN Breaks Down for Mortality Prediction
### Observed behavior during tuning
- GridSearchCV repeatedly selects very small k values, often collapsing to k = 1.
- weights='distance' is consistently chosen over weights='uniform'.
- Training scores appear artificially high, while test performance remains noticeably lower.

### Why k collapses to 1
- Under the F2-score, the model is pushed to maximize recall.
- Smaller k values classify borderline cases as positive, boosting recall inside cross-validation.
- KNN with small k becomes highly sensitive to noise and overfits the training folds.

### Why distance weighting dominates
- With weights='distance', a point’s closest neighbor is often itself, which receives overwhelming influence.
- This produces perfect or near-perfect training predictions across many k values.
- High training performance is therefore the result of memorization, not pattern learning.

### Clinical context that makes KNN unstable
- Mortality datasets are typically imbalanced, with relatively few positive cases.
- High-dimensional clinical features weaken KNN’s distance metric (curse of dimensionality).
- Local neighborhoods rarely reflect the true risk structure; majority-class votes dominate unless k is extremely small.
- The model compensates by shrinking k, which increases variance and destabilizes predictions.

### What this behavior indicates
- KNN cannot form a stable or generalizable decision boundary for this task.
- Its performance depends heavily on local data density and noise, making it unreliable for clinical prediction.
- Even when cross-validation appears strong for certain configurations, test-set performance exposes overfitting.

## Conclusion: KNN Is Not the Right Model

- The tuning behavior—k collapsing to 1, distance weighting dominating, and perfect training predictions—reveals that KNN is fundamentally incompatible with mortality prediction. 
- The combination of class imbalance, high dimensionality, and the need for calibrated, recall-sensitive predictions makes KNN unstable and prone to overfitting.
- A more appropriate approach is to replace KNN with models that handle these conditions effectively, such as logistic regression with class weights or gradient-boosted tree models, which provide stronger generalization and more reliable clinical performance.

ICD-9 Mortality Encoding: Handling Target Leakage in Practice

To improve the model’s understanding of patient comorbidity burden, I created features by estimating each ICD-9 code’s historical mortality risk and aggregating these risks per patient. This produces informative and clinically intuitive predictors.

The Issue

These ICD-9 mortality risks are computed from the target variable.

If I compute them using the full training data and then run cross-validation, each validation fold indirectly uses its own labels when generating features.

This leads to mild, within-training target leakage that inflates cross-validation scores.

How Problematic Is This?

Most ICD-9 codes appear many times, so removing a fold barely changes their estimated mortality rates.

The resulting optimism is small and nearly constant across folds.

Because the bias is uniform, model comparison and hyperparameter selection remain reliable.

Production Perspective

Real clinical pipelines typically maintain global risk tables derived from historical data (e.g., Charlson or Elixhauser weights).

Fold-specific encoders add complexity and provide limited benefit when the dataset is large enough for stable estimates.

My Practical Approach

I compute a global ICD-9 mortality mapping using the full training set.

I use this fixed mapping to generate comorbidity features for both train and test.

This avoids any leakage into the test set and keeps the feature pipeline stable, simple, and production-friendly.

Summary

I accept a small, uniform CV optimism in exchange for a cleaner and more maintainable workflow.

Hyperparameter tuning remains valid, test leakage is avoided, and the final model benefits from consistent, clinically meaningful comorbidity features.

Section: ICD-9 Mortality Encoding and Leakage

Approach 1: Global mortality mapping (simple, production-ready)

Compute mortality rates per ICD9 using all training data

Merge into train/test to create features

Pros: simple, stable, realistic for deployments

Cons: mild CV optimism due to target leakage

Result: CV score inflated by ~X%

Approach 2: Out-of-fold encoding (theoretically correct)

Compute mortality proxies separately for each fold

Encode validation folds using only fold-train data

Merge global mapping only for final test set

Pros: no leakage, unbiased CV

Cons: more engineering complexity

Result: CV score decreased by ~X% but hyperparameter ranking unchanged

Conclusion

Leakage did not affect test performance or model selection

Chose global mapping for production simplicity


During feature importance analysis, demographic variables such as ethnicity and religion appeared as moderately predictive. However, these features do not reflect physiological status and may introduce demographic bias. For ethical and governance reasons, I excluded these fields from the final model. Removing them had negligible impact on overall performance, demonstrating that the model’s predictive power is driven by clinical and comorbidity-related features rather than demographic information.