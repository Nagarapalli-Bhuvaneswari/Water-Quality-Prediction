import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import math

# Load the dataset and handle missing values
data = pd.read_csv('water_potability.csv')
data.fillna(data.mean(), inplace=True)

# Split the dataset into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split the dataset into training and test sets
split_ratio = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the Decision Tree model
#dt = DecisionTreeClassifier(random_state=42)
#dt.fit(X_train, y_train)
#y_pred = dt.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy * 100:.2f}%")


from sklearn.model_selection import GridSearchCV

# Define hyperparameters to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a Decision Tree classifier with the best hyperparameters
dt = DecisionTreeClassifier(random_state=42, **best_params)

# Train the model with the best hyperparameters
dt.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def get_dt_accuracy():
    return f"{math.ceil(accuracy_score(y_test,y_pred)*100)}"
