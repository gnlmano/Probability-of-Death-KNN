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


