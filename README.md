# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.

2.Calculate the mean of the X -values and the mean of the Y -values.

3.Find the slope m of the line of best fit using the formula.

<img width="296" height="134" alt="Screenshot 2026-04-24 142329" src="https://github.com/user-attachments/assets/c84b8915-a172-4d3c-b25a-6c72aca3d759" />

4. Compute the y -intercept of the line by using the formula:

<img width="209" height="51" alt="Screenshot 2026-04-24 142402" src="https://github.com/user-attachments/assets/41200e24-f859-421e-89b0-948737a601a0" />

5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Megala M S
RegisterNumber: 212225040230
*/
# Implementation of Random Forest Algorithm
# Weather Prediction Model

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Create Dataset
data = {
    "Temperature": [30, 32, 35, 28, 25, 27, 33, 31, 29, 26],
    "Humidity": [70, 65, 60, 80, 90, 85, 55, 68, 75, 88],
    "WindSpeed": [10, 12, 8, 15, 20, 18, 7, 11, 14, 19],
    "Weather": [
        "Sunny", "Sunny", "Sunny", "Rainy", "Rainy",
        "Rainy", "Sunny", "Sunny", "Rainy", "Rainy"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Convert Categorical Labels to Numeric
df["Weather"] = df["Weather"].map({
    "Sunny": 1,
    "Rainy": 0
})

# Features and Target
X = df[["Temperature", "Humidity", "WindSpeed"]]
y = df["Weather"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Evaluation:")

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

# Step 9: Visualization of Feature Importance
plt.figure(figsize=(8,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")

plt.grid(True)

plt.show()

# Step 10: Custom Prediction
weather_data = [[29, 72, 13]]

prediction = model.predict(weather_data)

print("\nCustom Weather Prediction:")

if prediction[0] == 1:
    print("Predicted Weather: Sunny")
else:
    print("Predicted Weather: Rainy")
```

## Output:
<img width="857" height="462" alt="image" src="https://github.com/user-attachments/assets/a1ee383c-f558-45a2-8129-6d17cc6091ed" />
<img width="1211" height="365" alt="image" src="https://github.com/user-attachments/assets/ec062e45-9e3c-484f-a49f-025bae070e97" />
<img width="1017" height="623" alt="image" src="https://github.com/user-attachments/assets/98ca5696-4891-4ce3-b5c9-baa185a12cf0" />
<img width="397" height="113" alt="image" src="https://github.com/user-attachments/assets/47f1abbd-47ea-415f-97f7-92a370b7eea0" />


## Result:
Thus the Implemented of Random Forest Algorithm for Weather Prediction using python programming.

