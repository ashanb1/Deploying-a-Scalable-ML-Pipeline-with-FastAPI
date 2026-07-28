# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier from scikit-learn's `RandomForestClassifier`, trained with default hyperparameters and `random_state=42` for reproducibility. It was developed as part of a machine learning pipeline deployed with FastAPI. The model was trained using scikit-learn's `ensemble` module, with categorical features one-hot encoded and the target label binarized prior to training.

## Intended Use
This model is intended to predict whether an individual's annual income exceeds $50K based on U.S. Census demographic and employment data. It is intended for educational and demonstration purposes, showing an end-to-end ML pipeline including model training, testing, and deployment via a REST API. It is not intended for use in real-world decision-making that affects individuals, such as lending, hiring, or eligibility determinations.

## Training Data
The training data comes from the publicly available Census Bureau "Adult" dataset (`census.csv`), which contains demographic and employment information collected from the 1994 U.S. Census. Features include age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week, and native country. The full dataset was split, with 80% used for training via `train_test_split` (`test_size=0.20`, `random_state=42`).

## Evaluation Data
The evaluation data is the remaining 20% of the census dataset held out from the training split (same `train_test_split` call). The same one-hot encoder and label binarizer fit on the training data were applied to the test data (with `training=False`) to ensure consistent preprocessing.

## Metrics
The model was evaluated using precision, recall, and F1 score (fbeta with beta=1). On the held-out test set, the model achieved:

- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

Performance was also computed on slices of the data for each categorical feature (see `slice_output.txt`). Performance varies notably across slices — for example, workclass categories with more data (e.g., "Private") show more stable metrics, while categories with very few samples (e.g., "Without-pay", count of 4) show extreme values (Precision/Recall/F1 of 1.0) that are not statistically reliable given the small sample size.

## Ethical Considerations
This dataset reflects historical income and demographic patterns from 1994 U.S. Census data, which may encode and perpetuate existing societal biases related to race, sex, and national origin. The slice-based performance analysis shows the model does not perform uniformly across all demographic groups, meaning predictions may be less reliable or more biased for underrepresented groups in the training data. This model should not be used to make real decisions about individuals, particularly in contexts like employment, credit, or housing, without a much more thorough fairness and bias evaluation.

## Caveats and Recommendations
- The dataset is from 1994 and does not reflect current economic conditions, wage levels, or demographic distributions.
- Some categorical slices have very small sample sizes, making their reported metrics unreliable (e.g., "Without-pay" workclass with only 4 test samples).
- No hyperparameter tuning was performed; the model uses scikit-learn's default `RandomForestClassifier` settings aside from `random_state`.
- Future improvements could include cross-validation instead of a single train/test split, hyperparameter tuning, and a deeper fairness audit across sensitive attributes (race, sex, native country) before any real-world use.