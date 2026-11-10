# (010000) Machine Learning - Spring 2026

Coursework for Machine Learning at Sejong University (Prof. Cheol Jeong). The course starts with Google Colab and Python basics, then moves through regression (linear, regularized, polynomial), feature engineering, cross-validation, hyperparameter tuning, and classification (logistic regression, decision trees, SVMs). It finishes with PCA and K-means clustering. There is a coding midterm on regularized regression and a coding final on PCA and SVMs.

All work is in Jupyter notebooks, written to run in Google Colab.

## Repository Structure

| Folder | Topic | Key concepts |
|---|---|---|
| [`01-google-colab`](./01-google-colab) | Exercise 1: Google Colab quickstart | Loading a CSV with pandas, `head`, `size`, `mean` |
| [`02-python-basics-1`](./02-python-basics-1) | Exercise 2: Python basics 1 | Data cleaning, slicing, `zip`, `enumerate`, comprehensions |
| [`03-python-basics-2`](./03-python-basics-2) | Exercise 3: Python basics 2 | Functions, default arguments, CSV reading and writing |
| [`04-linear-regression`](./04-linear-regression) | Exercise 4: Linear regression | Single-feature regression, MSE, feature comparison |
| [`05-multiple-linear-regression`](./05-multiple-linear-regression) | Exercise 5: Multiple input features | Multiple regression, residual analysis |
| [`06-regularized-regression`](./06-regularized-regression) | Exercise 6: Regularized regression | Ridge, Lasso, choosing `alpha` |
| [`07-polynomial-regression`](./07-polynomial-regression) | Exercise 7: Polynomial regression | Polynomial features, 3D regression surface |
| [`08-feature-engineering`](./08-feature-engineering) | Exercise 8: Feature engineering | Interaction features, train/validation comparison |
| [`09-cross-validation`](./09-cross-validation) | Exercise 9: Cross-validation | Data leakage, hold-out vs. k-fold |
| [`10-hyperparameter-tuning`](./10-hyperparameter-tuning) | Exercise 10: Hyperparameter tuning | Pipelines, `GridSearchCV` |
| [`11-decision-tree`](./11-decision-tree) | Exercise 11: Decision tree | Entropy criterion, feature importance |
| [`12-support-vector-machine-1`](./12-support-vector-machine-1) | Exercise 12: SVM, part 1 | Linear SVM, effect of `C`, margins, support vectors |
| [`13-support-vector-machine-2`](./13-support-vector-machine-2) | Exercise 13: SVM, part 2 | Linear, polynomial and RBF kernels |
| [`14-pca-classification`](./14-pca-classification) | Exercise 14: PCA-based classification | PCA, explained variance, PCA + logistic regression |
| [`15-k-means-clustering`](./15-k-means-clustering) | Exercise 15: K-means clustering | Elbow curve, silhouette score, choosing K |
| [`practice`](./practice) | In-class practice notebooks (01 to 04, 12 to 21) | Colab, Python basics, regression, classification, missing data, PCA, K-means |
| [`midterm-exam`](./midterm-exam) | Coding midterm (template plus two attempts) | ElasticNet, `ColumnTransformer`, grid search |
| [`final-exam`](./final-exam) | Coding final exam | PCA, RBF SVM, grid search, decision region |


## Assignments

### 01 - Google Colab Quickstart
**Purpose:** Get comfortable with Colab and the basic pandas workflow.
**Description:** Uploads and loads `Car_Price_Prediction.csv` (columns: Make, Model, Year, Engine Size, Mileage, Fuel Type, Transmission, Price). The notebook previews the data with `head()`, checks `df.size` (8,000 values), and takes the mean of each numeric column. Average price is about $25,137. The handout is a short list of steps ending with submission to the course site.

### 02 - Python Basics 1
**Purpose:** Practice the Python tools used throughout the course.
**Description:** Cleans a messy list of image filenames and labels. Steps: strip whitespace, lowercase the names, keep only `.jpg` and `.jpeg` files, and drop empty entries. It then builds label-to-index mappings, takes slices, finds positions with `enumerate`, and computes a character count, using list comprehensions throughout. The five surviving pairs match the expected output.

### 03 - Python Basics 2
**Purpose:** Practice functions and file I/O.
**Description:** Fill-in-the-blank exercise. A function adds a prefix to each filename (with a default prefix argument in the solution), `enumerate` numbers the results, and the `csv` module writes them to a file and reads them back as dictionaries.
**Note on my version:** The CSV is written without a header row, and the `DictReader` field name is typed `'No,'` instead of `'No.'`, so the printed keys do not match the expected output. The solution writes a header row and reads it back directly.

### 04 - Linear Regression
**Purpose:** Fit single-feature linear regressions and pick the best by MSE.
**Description:** Using the Boston-style housing dataset (`Housing price dataset_modified.csv`, 506 rows), choose the best single feature among `INDUS`, `DIS` and `PTRATIO` for predicting `MEDV`, plot the fit, and report the MSE. The solution finds `PTRATIO` best (MSE about 62.7, versus 64.7 for `INDUS` and 79.1 for `DIS`).
**Note on my version:** It also picks `PTRATIO`, but it regresses each feature on `MEDV` instead of `MEDV` on each feature, so the MSE values (3.5 to 36.0) are on a different scale. It also uses `df`, `a_pred` and `b_pred`, which are not defined in the saved cells, so it will not run from top to bottom without edits.

### 05 - Linear Regression with Multiple Input Features
**Purpose:** Extend regression to several inputs and check the residuals.
**Description:** Choose two features that are negatively correlated with `MEDV`, fit a multiple linear regression, plot a histogram of the residuals, and discuss whether they look Gaussian. My version uses `TAX` and `NOX` and concludes the residuals do not look Gaussian. The solution uses `CRIM` and `NOX`, chosen from a correlation heatmap, and also adds predicted-vs-true and residual plots.

### 06 - Regularized Linear Regression
**Purpose:** Compare Ridge and Lasso across regularization strengths.
**Description:** Standardizes the housing features, uses 100 training and 100 validation samples, and trains Ridge and Lasso for `alpha` in 0.01, 0.1, 1, 10 and 100. Training and validation MSE are compared. Ridge gives its best validation MSE at `alpha = 100` (about 34.4). Lasso is best at `alpha = 0.01` (about 35.2), and at `alpha` of 10 or more its MSE is identical, consistent with all coefficients being shrunk to zero. My results and conclusions match the solution.

### 07 - Polynomial Regression
**Purpose:** Fit a nonlinear model by expanding the features.
**Description:** Predicts `MEDV` from `RM` and `INDUS` with an order-2 polynomial regression on the whole dataset (no split). The MSE is about 32.1 in both my version and the solution. The notebook also draws an interactive 3D plot of the data and regression surface with `plotly`.
**Note on my version:** The axis titles in the 3D plot are swapped, because the x-axis holds `RM` but is labeled `INDUS`.

### 08 - Feature Engineering
**Purpose:** See whether engineered features improve a linear model.
**Description:** Uses five features (`INDUS`, `RM`, `DIS`, `AGE`, `NOX`) with a 60/20/20 train/validation/test split. A baseline linear regression is compared with a model where one feature is replaced by an interaction of two features. My version replaces `NOX` with `NOX × RM` (an attempt with `NOX` and `AGE` did not lower the MSE), which cuts validation MSE from about 32.4 to 30.8. The solution instead adds an `RM × NOX` feature in place of `AGE`.

### 09 - Cross-validation
**Purpose:** Find and fix data leakage, and compare evaluation methods.
**Description:** On scikit-learn's diabetes dataset, a fixed-alpha Ridge model is evaluated by a hold-out split and by 5-fold cross-validation. The leakage fix is to split first and fit the scaler on training data only, including inside each CV fold. Hold-out validation MSE is about 3,271, the 5-fold mean is about 2,986 (std about 587), and the final test MSE is about 3,148.
**Note on my version:** The corrected code produces the same numbers as the solution, but the notebook has no written explanation of where the leakage occurred, which the handout asked for.

### 10 - Hyperparameter Tuning
**Purpose:** Tune several hyperparameters at once without leaking test data.
**Description:** Builds a pipeline of `StandardScaler`, `PolynomialFeatures` and `Lasso`, then runs `GridSearchCV` (5-fold) over polynomial degree 1 to 3 and Lasso `alpha` from 0.001 to 10, which is 15 combinations. The solution's best setting is degree 2 with `alpha = 0.01` (CV MSE about 19.0), with a final test MSE of about 19.0.
**Note on my version:** I set `Lasso(tol=0.1)` to quiet convergence warnings. It chose degree 2 with `alpha = 0.1` and got a test MSE of about 23.2, probably because of the looser tolerance.

### 11 - Decision Tree
**Purpose:** Train and interpret a decision tree classifier.
**Description:** Switch Practice 15 from Iris to the Wine dataset and from the `gini` criterion to `entropy`. With `max_depth=4`, a 70/30 stratified split and `random_state=2026`, report the confusion matrix, classification report and a feature-importance bar chart. The tree gets about 99% training accuracy and 81% test accuracy.
**Note on my version:** The confusion matrix and classification report match the solution, but there is no feature-importance bar chart, and some comments still mention the Iris dataset.

### 12 - Support Vector Machine, Part 1
**Purpose:** See how the regularization parameter `C` changes a linear SVM.
**Description:** On the Wine dataset, using classes 0 and 1 and the features `alcohol` and `malic_acid`, standardize the features and train two linear SVMs with `C = 100` and `C = 0.1`. Report training and test accuracy and support vectors per class, then plot the decision boundary, margins and support vectors for both. In the solution, `C = 100` gives 91.2% training and 97.4% test accuracy with 22 support vectors, and `C = 0.1` gives 90.1% and 94.9% with 41 support vectors: a smaller `C` means a wider margin and more support vectors.
**Note on my version:** The code selects classes 1 and 2 instead of 0 and 1, so both models score about 88% training and 72% test accuracy, and the plot labels do not match the data.

### 13 - Support Vector Machine, Part 2
**Purpose:** Compare SVM kernels on a multi-class problem.
**Description:** On all 13 Wine features and 3 classes (stratified 70/30 split, `random_state=520`), train three `StandardScaler` + `SVC` pipelines with `C = 10`: linear, degree-3 polynomial and RBF. The solution compares test accuracy and support-vector counts, then shows the confusion matrix and classification report for the best model. RBF wins with 98.2% test accuracy (linear and polynomial both get 94.4%).
**Status:** Instructor solution only so far.

### 14 - PCA-based Classification
**Purpose:** Use PCA to reduce dimensions before classifying.
**Description:** On the Wine dataset (13 features, 3 classes), standardize the training set and fit PCA. The exercise asks for a scree plot and cumulative explained variance, the number of components needed for 90% and 95% of the variance, a 2D PCA scatter plot, and a `StandardScaler` → `PCA` → `LogisticRegression` pipeline with 2 components. Finally it asks for a confusion matrix and classification report for the best pipeline. In the solution, the 2-component pipeline gets 97.6% training and 94.4% test accuracy.
**Note on my version:** The scree plot, summary table and 2-component pipeline are done (about 98% train and 94% test). However, the 2D scatter plot and the confusion matrix and report reuse labels from a two-class example (`negative` and `positive`), so the third class is missing and the report should fail with three classes. The matrix title also says "RF" instead of logistic regression.

### 15 - K-means Clustering
**Purpose:** Choose the number of clusters using the elbow method and silhouette score.
**Description:** On the Iris dataset, standardize the features and run K-means for K from 2 to 8, recording distortion and silhouette score. Plot the elbow and silhouette curves, choose the K with the highest silhouette, and visualize the clusters on the first two principal components. The solution picks K = 2 (silhouette about 0.58, distortion about 222.4, cluster sizes 50 and 100).
**Note on my version:** It uses the Wine dataset instead of Iris, so its numbers differ (silhouette peaks at K = 3, about 0.28). The structure otherwise follows the handout.

## Practice Notebooks (`practice/`)

In-class notebooks, numbered to match the class sessions. Practice 05 to 11 are not in this repo.

| Notebook | Purpose and description |
|---|---|
| `practice_01_google_colab` | Colab basics: markdown cells, `nvidia-smi`, checking CUDA with PyTorch, uploading files, mounting Google Drive, and loading the Car Price CSV. |
| `practice_02_python_basics_1` | Python basics: variables, control flow, functions and data structures. |
| `practice_03_python_basics_2` | Exceptions, file I/O, CSV and JSON, NumPy, pandas and matplotlib. |
| `practice_04_linear_regression` | Linear regression on the Housing dataset. |
| `practice_12_classification` | Binary logistic regression on Iris (versicolor vs. virginica): scatter plots, stratified 70/30 split, `StandardScaler` + `LogisticRegression` pipeline, predicted probabilities, confusion matrix and classification report. |
| `practice_13_multi_class_classification` | Multi-class logistic regression on all three Iris classes (`random_state=1234`). |
| `practice_13_my_work` | My own version of Practice 13 (Iris, scaler + logistic regression pipeline). **Note:** It replicates the handout, has typos in some printed strings ("Input Shap", "First Fice") and ends with an empty cell. |
| `practice_14_knn_classification` | K-nearest neighbors on the Wine dataset. |
| `practice_15_decision_tree` | Decision tree on Iris (`gini`, `max_depth=3`). Exercise 11 builds on this. |
| `practice_16_svm_1` | Linear SVM on Iris, comparing a large and a small `C`. |
| `practice_17_svm_2` | SVM kernels on Iris: the effect of `gamma`, grid search, and multi-class SVM. |
| `practice_18_missing_data_handling` | Hepatitis dataset (OpenML): compare `dropna`, `fillna` and `SimpleImputer`. |
| `practice_19_random_forest` | Random forest on the Pima diabetes dataset (OpenML). |
| `practice_20_pca_classification` | PCA pipeline on the Pima diabetes data. It reads `pima-indians-diabetes.csv`, which is not included here (see Datasets), and treats zeros in `plas`, `pres`, `skin`, `insu` and `mass` as missing values to impute with the median. |
| `practice_21_k_means_clustering` | K-means clustering on the Wine dataset. |


**Status of my attempt:** Incomplete. The feature-group lists and the data split are done, and the preprocessing pipeline was built. Remaining problems: `"M_job"` is a typo for `"Mjob"`, the ordinal encoder uses quality categories (`Po`, `Fa`, `TA`, `Gd`, `Ex`) from another dataset, and the grid-search cell has a syntax error and tunes the wrong parameter names. The MSE cells and the final save cell never run because they use undefined or empty variables.
**Files:** `midterm_exam_template.ipynb` is the blank exam, `midterm_exam_my_attempt.ipynb` is my first attempt, and `midterm_exam_my_attempt_v2.ipynb` is a later attempt made from the same template. The later one is still incomplete: it has the same ordinal-encoder and `M_job` problems, the `param_grid` has a missing comma and uses `lasso__alpha`, it calls `.fit` on `param_grid`, and the last cell stops at half-written lines (`grid.best_piepline`, `dev_pred = grid.`).


## Datasets

The data files are not included in this repository. Upload them to Colab (or place them next to the notebook) to run the notebooks:

- `Car_Price_Prediction.csv` (Exercise 1)
- `Housing price dataset_modified.csv` (Exercises 4 to 10)
- `rice_cammeo_osmancik_mod.csv` (final exam)
- `pima-indians-diabetes.csv` (Practice 20)

Exercises 9 and 11 to 15 and most practice notebooks use datasets that ship with scikit-learn (diabetes, Wine, Iris). Practices 18 and 19 download the Hepatitis and Pima diabetes data from OpenML with `fetch_openml`. The midterm downloads the Student Performance dataset from the UCI repository (id 320) with `ucimlrepo`.

## Running the Notebooks

1. Open a notebook in [Google Colab](https://colab.research.google.com/) (*File > Upload notebook*).
2. Upload any needed CSV to the Colab session.
3. Choose *Runtime > Run all*.

To run locally, install `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly` and `ucimlrepo`, then open the notebooks in Jupyter. Exercise 1 imports `google.colab`, which only works in Colab.

## Tools

- Python 3
- pandas, NumPy
- scikit-learn (regression, pipelines, `GridSearchCV`, PCA, SVM, decision trees, K-means)
- matplotlib, seaborn, plotly
- Google Colab
