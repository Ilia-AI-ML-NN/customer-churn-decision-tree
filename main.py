""" Customer Churn Prediction with Decision Tree Classifier """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn import tree

# ----------------------
# 1. Загрузка данных
# ----------------------
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

# ----------------------
# 2. Предобработка данных
# ----------------------
def preprocess(df):
    df = df.drop("customerID", axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    x = df.drop('Churn', axis=1)
    y = df['Churn']
    return x, y

# ----------------------
# 3. Обучение модели и оценка
# ----------------------
def train_and_evaluate(x, y, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Сохраняем модель
    joblib.dump(model, 'decision_tree_model.joblib')

    # Метрики
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, x_test, y_test, y_pred

# ----------------------
# 4. Визуализация дерева
# ----------------------
def visualize_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    tree.plot_tree(model, feature_names=feature_names, class_names=['No Churn','Churn'], filled=True)
    plt.show()

# ----------------------
# 5. Визуализация матрицы ошибок
# ----------------------
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ----------------------
# 6. Основной блок
# ----------------------
if __name__ == "__main__":
    df = load_data()
    x, y = preprocess(df)

    # Обучение и оценка модели
    model, x_test, y_test, y_pred = train_and_evaluate(x, y)

    # Матрица ошибок
    plot_confusion(y_test, y_pred)

    # Визуализация дерева
    visualize_tree(model, x.columns)
