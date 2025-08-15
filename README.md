# Customer Churn Prediction with Decision Tree Classifier

## Project Description
This project predicts customer churn for a telecom company using a **Decision Tree Classifier**.  
It provides accuracy, precision, recall, F1-score metrics, as well as a tree visualization and a confusion matrix.

---

## Data
The dataset is from an open-source IBM dataset:  
[TELCO Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

---

## Installation

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt


Usage

Run the main script:

python churn_decision_tree.py


The script performs:

Data loading and preprocessing

Training a Decision Tree model

Printing metrics (Accuracy, Precision, Recall, F1-score)

Plotting the confusion matrix

Visualizing the decision tree

Saving the trained model to decision_tree_model.joblib

To reuse the saved model:

import joblib
model = joblib.load('decision_tree_model.joblib')

Visualizations

Confusion Matrix shows how well the model classifies customers correctly and incorrectly.

Decision Tree helps visually understand which features influence customer churn.
