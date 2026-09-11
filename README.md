# The Predictive Performance Audit Console & Pipeline Packaging

## Week 4 — Final Milestone

This project implements a complete predictive performance audit pipeline for supervised machine learning classification models.

The main objective of Week 4 is to evaluate machine learning models beyond simple accuracy by using multiple evaluation metrics, confusion matrices, ROC-AUC analysis, cross-validation, and overfitting analysis.

The project also consolidates the complete workflow into an executable and modular pipeline.

---

## Project Objective

The **Predictive Performance Audit Console & Pipeline Packaging** evaluates the performance and robustness of supervised classification algorithms.

The project focuses on:

- Classification metric evaluation
- Precision, Recall, F1-Score, and ROC-AUC
- Confusion matrix analysis
- ROC curve comparison
- 5-fold cross-validation
- Overfitting detection
- Bias-variance analysis
- Model stability analysis
- End-to-end pipeline execution
- Modular implementation
- Performance comparison and trade-offs

---

## Week 4 Requirements

This project satisfies the required Week 4 final milestone tasks.

### 1. Evaluation Metric Engine

The classification models are evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Training Accuracy
- Testing Accuracy

F1-Score is particularly important because the target variable is imbalanced.

---

### 2. Confusion Matrix and Error Distribution

Confusion matrices are generated for both classification algorithms.

They provide information about:

- True Negatives
- False Positives
- False Negatives
- True Positives

ROC curves are also generated to compare the ranking performance of the two models.

---

### 3. Cross-Validation and Overfitting Audit

Stratified **5-fold cross-validation** is performed using F1-Score.

The cross-validation process measures:

- Performance across validation folds
- Mean F1-Score
- Standard deviation of F1-Score
- Model stability

An overfitting audit compares:

- Training Accuracy
- Testing Accuracy
- Train-Test Gap
- Cross-Validation Mean F1
- Cross-Validation Standard Deviation

---

### 4. End-to-End Pipeline Consolidation

The complete workflow is consolidated into the main execution script.

```text
Data Cleaning
      ↓
Feature Engineering
      ↓
Training
      ↓
Metric Audit
      ↓
Cross-Validation
      ↓
Overfitting Analysis

The main execution script is:

main.py

The implementation is divided into modular source files for model training, evaluation, and cross-validation.

5. Documentation

This README documents:

Dataset details
Algorithms used
Evaluation metrics
Model performance
Cross-validation results
Overfitting analysis
Model trade-offs
Generated outputs
Project structure
Replication instructions
Dataset

The Week 4 project uses the feature matrix and target variable generated during the previous supervised learning stage.

Property	Value
Total Records	15,000
Features	61
Target Variable	Churn
Training Records	12,000
Testing Records	3,000
Missing Values	0
Duplicate Rows	0
Target Distribution
Churn Class	Records
0	12,702
1	2,298

The target distribution is imbalanced, so F1-Score is used for cross-validation to provide a more meaningful evaluation of the churn classification performance.

Algorithms Evaluated

Two supervised classification algorithms are evaluated:

Logistic Regression
Decision Tree Classifier

Both models are trained using the same training and testing datasets to provide a fair performance comparison.

Performance Comparison

The final testing performance of both algorithms is shown below.

Model	Training Accuracy	Testing Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.8473	0.8447	0.4800	0.1565	0.2361	0.8197
Decision Tree	1.0000	0.8560	0.5306	0.5283	0.5294	0.7218
5-Fold Cross-Validation Results

Stratified 5-fold cross-validation was performed using F1-Score.

Model	Fold 1	Fold 2	Fold 3	Fold 4	Fold 5	Mean F1	Std F1
Logistic Regression	0.2532	0.2467	0.2675	0.2744	0.2508	0.2585	0.0106
Decision Tree	0.4989	0.4709	0.5590	0.5420	0.5157	0.5173	0.0311
Overfitting Audit
Model	Training Accuracy	Testing Accuracy	CV Mean F1	CV Std F1	Train-Test Gap
Logistic Regression	0.8473	0.8447	0.2585	0.0106	0.0026
Decision Tree	1.0000	0.8560	0.5173	0.0311	0.1440
Bias-Variance and Generalization Analysis
Logistic Regression

Logistic Regression has a very small train-test performance gap of 0.0026 and a low cross-validation standard deviation of 0.0106.

This indicates stable performance across the training and testing datasets and relatively consistent behavior across validation folds.

However, its recall and F1-Score for the Churn class are relatively low.

Decision Tree

The Decision Tree achieves higher testing accuracy, precision, recall, and F1-Score than Logistic Regression.

However, its training accuracy is 1.0000 while testing accuracy is 0.8560, resulting in a train-test gap of 0.1440.

This large gap indicates a strong overfitting tendency.

The Decision Tree also has a higher cross-validation standard deviation of 0.0311, indicating greater variation across validation folds.

Model Trade-Offs

The two algorithms demonstrate different strengths.

Aspect	Logistic Regression	Decision Tree
Testing Accuracy	Lower	Higher
Precision	Lower	Higher
Recall	Lower	Higher
F1-Score	Lower	Higher
ROC-AUC	Higher	Lower
Train-Test Gap	Very small	Large
CV Variation	Lower	Higher
Generalization Stability	More stable	More variable
Overfitting	Low tendency	Strong tendency
Overall Finding

The Decision Tree provides better testing F1-Score and recall, making it stronger at the selected classification threshold.

However, Logistic Regression achieves a higher ROC-AUC and demonstrates more stable generalization.

Therefore, the results show a clear trade-off between classification performance at the selected threshold and overall ranking performance across thresholds.

Confusion Matrix Results

Confusion matrices provide detailed information about the specific classification errors made by each model.

Logistic Regression
[[2462   78]
 [ 388   72]]
Interpretation
True Negatives: 2462
False Positives: 78
False Negatives: 388
True Positives: 72

Decision Tree
[[2325  215]
 [ 217  243]]
Interpretation
True Negatives: 2325
False Positives: 215
False Negatives: 217
True Positives: 243

ROC-AUC Results
Model	ROC-AUC
Logistic Regression	0.8197
Decision Tree	0.7218

Logistic Regression achieves the higher ROC-AUC score, indicating stronger overall ranking ability across classification thresholds.

ROC Curve Comparison

Generated Outputs
Reports

The pipeline generates the following reports:

outputs/reports/
├── model_metrics.csv
├── cross_validation_results.csv
└── performance_audit.csv
model_metrics.csv

Contains:

Training Accuracy
Testing Accuracy
Precision
Recall
F1-Score
ROC-AUC
cross_validation_results.csv

Contains:

Five validation fold F1-Scores
Mean F1-Score
Standard deviation of F1-Score
performance_audit.csv

Contains:

Training Accuracy
Testing Accuracy
CV Mean F1
CV Standard Deviation
Train-Test Gap
Generated Figures

The pipeline generates:

outputs/figures/
├── logistic_regression_confusion_matrix.png
├── decision_tree_confusion_matrix.png
└── roc_curve_comparison.png
Project Structure
the-predictive-performance-audit-console/
│
├── data/
│   ├── final_feature_matrix.csv
│   └── target_churn.csv
│
├── outputs/
│   ├── figures/
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   └── roc_curve_comparison.png
│   │
│   └── reports/
│       ├── cross_validation_results.csv
│       ├── model_metrics.csv
│       └── performance_audit.csv
│
├── src/
│   ├── cross_validation.py
│   ├── evaluation.py
│   └── models.py
│
├── notebooks/
│   └── Week4_Predictive_Performance_Audit.ipynb
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
Modular Implementation

The project follows a modular implementation structure.

Model Training Module

Model training functions are implemented in:

src/models.py

The module contains dedicated functions for:

Logistic Regression training
Decision Tree training
Evaluation Module

Model evaluation functions are implemented in:

src/evaluation.py

The module handles:

Precision
Recall
F1-Score
ROC-AUC
Confusion matrices
ROC curve generation
Model metric comparison
Cross-Validation Module

Cross-validation functions are implemented in:

src/cross_validation.py

The module performs:

Stratified 5-fold cross-validation
Fold-level F1-Score calculation
Mean F1-Score calculation
Standard deviation calculation
Main Pipeline

The main pipeline is implemented in:

main.py

The main script handles:

Feature matrix loading
Target variable loading
Dataset verification
Train-test splitting
Logistic Regression training
Decision Tree training
Classification metric calculation
Model comparison
Confusion matrix generation
ROC curve generation
5-fold cross-validation
Overfitting audit
Report generation
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook
Google Colab
Git
GitHub
How to Run the Project
1. Clone the Repository
git clone https://github.com/saif-m7/the-predictive-performance-audit-console.git
2. Navigate to the Project
cd the-predictive-performance-audit-console
3. Create a Virtual Environment

For Windows PowerShell:

python -m venv venv
4. Activate the Virtual Environment
.\venv\Scripts\Activate.ps1
5. Install Required Dependencies
pip install -r requirements.txt
6. Execute the Pipeline
python main.py

The pipeline will:

Load the feature matrix and target variable
Verify the dataset
Train both models
Calculate evaluation metrics
Generate confusion matrices
Generate the ROC curve
Perform 5-fold cross-validation
Complete the overfitting audit
Save the generated reports
Google Colab Notebook

A publicly viewable Google Colab notebook is provided for quick execution and verification.

Open Week 4 Google Colab Notebook

Open Week 4 Google Colab Notebook

The notebook demonstrates:

Data loading
Dataset verification
Train-test splitting
Logistic Regression training
Decision Tree training
Classification metrics
Confusion matrix analysis
ROC curve comparison
5-fold cross-validation
Overfitting audit
Performance interpretation
Report generation
Figure generation
GitHub Repository

Open GitHub Repository

Technical Reference Material

The following technical references were used to support the implementation:

Scikit-learn Model Evaluation Documentation
Scikit-learn Cross-Validation Documentation
Google Machine Learning — Classification Metrics
Final Conclusion

The Predictive Performance Audit Console successfully evaluates supervised classification models beyond simple accuracy.

The project implements the required:

Evaluation metric engine
Precision, Recall, F1-Score, and ROC-AUC analysis
Confusion matrix analysis
ROC curve comparison
5-fold cross-validation
Overfitting audit
Bias-variance analysis
End-to-end pipeline consolidation
Modular source-code organization
Performance comparison and trade-off analysis

Logistic Regression demonstrates more stable generalization and higher ROC-AUC, while the Decision Tree achieves better testing F1-Score and recall but exhibits a significant overfitting tendency.

The project demonstrates the importance of using multiple evaluation metrics and cross-validation when assessing predictive performance and model robustness.

Week 4 Final Milestone Completed Successfully

This project fulfills the requirements of the GetIntern Week 4 Final Milestone: The Predictive Performance Audit Console & Pipeline Packaging.