import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width",
                "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, header=None, names=column_names)
# print(iris_data)

# Preprocessing data
# bagi data train (80%) dan test (20%)

X = iris_data.iloc[:, :-1]
y = iris_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Bangun model K-NN dengan K=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model
model_path = os.path.join(current_dir, 'iris_model.pkl')
joblib.dump(model, model_path)
