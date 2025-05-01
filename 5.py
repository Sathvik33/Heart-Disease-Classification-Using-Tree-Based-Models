import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.environ["GRAPHVIZ_BINARY"] = r"C:\Graphviz-12.2.1-win64\bin\dot.exe"

# Path setup
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

# Load data
data = pd.read_csv(DATA_PATH)

# Label Encoding for categorical columns (if any)
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop(columns=["target"])
y = data["target"]

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=46)

# Train Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize Decision Tree

dot_data = export_graphviz(dt,
                           out_file=None,
                           feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render(os.path.join(BASE_DIR, "output", "decision_tree_heart"), format="png", cleanup=True)
graph.view()


#Analyzing Overfitting with Decision Tree

train_scores = []
test_scores = []
depths = range(1, 21)  # Try tree depths from 1 to 20

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(x_train, y_train)
    
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, marker='o', label='Training Accuracy')
plt.plot(depths, test_scores, marker='s', label='Testing Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Overfitting Analysis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Train Random Forest Classifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# Feature Importances (Random Forest)
importances = rf.feature_importances_
features = data.drop(columns=["target"]).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Cross-Validation
dt_scores = cross_val_score(dt, X_scaled, y, cv=5)
rf_scores = cross_val_score(rf, X_scaled, y, cv=5)

print("\nCross-Validation Scores:")
print(f"Decision Tree CV Accuracy: {dt_scores.mean()*100:.2f}")
print(f"Random Forest CV Accuracy: {rf_scores.mean()*100:.2f}")

