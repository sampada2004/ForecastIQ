'''import pandas as pd 
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree  # Importing plot_tree for decision tree visualization
from sklearn.metrics import r2_score  # Importing R^2 score calculation

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data = data.dropna()

# Plotting scatter plot using Plotly Express
fig = px.scatter(data, x="Units Sold", y="Total Price", size="Units Sold")
fig.show()

# Prepare data for modeling
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model with max_depth=3 (smaller tree for readability)
model = DecisionTreeRegressor(max_depth=3)

# Train the model
model.fit(xtrain, ytrain)

# Make predictions on a new data point (features)
features = np.array([[133.00, 140.00]])
print("Predicted Units Sold:", model.predict(features))

# Predict on test data
ypred = model.predict(xtest)

# Calculate R^2 score on test data
r2 = r2_score(ytest, ypred)
print("R^2 Score on test data:", r2)

# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=["Total Price", "Base Price"], rounded=True)
plt.title("Decision Tree Regression (Max Depth = 3)")
plt.show()
'''
import pandas as pd 
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
data = data.dropna()

# Plotting scatter plot using Plotly Express
fig = px.scatter(data, x="Units Sold", y="Total Price", size="Units Sold")
fig.show()

# Prepare data for modeling
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model
model = DecisionTreeRegressor(max_depth=3,random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', verbose=1)
grid_search.fit(xtrain, ytrain)

print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_
model.fit(xtrain, ytrain)
# Function to predict Units Sold based on user input
def predict_units_sold(total_price, base_price):
    features = np.array([[total_price, base_price]])
    predicted_units_sold = best_model.predict(features)[0]  #best_model instead of model
    return predicted_units_sold

# Example usage:
total_price_input = float(input("Enter Total Price: "))
base_price_input = float(input("Enter Base Price: "))

predicted_units = predict_units_sold(total_price_input, base_price_input)
print("Predicted Units Sold:", int(round(predicted_units)))
plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=["Total Price", "Base Price"], rounded=True)
plt.title("Decision Tree Regressor")
plt.show()
# Predict on test data
ypred = best_model.predict(xtest) #best_model instead of model

# Calculate R^2 score on test data
r2 = r2_score(ytest, ypred)
print("R^2 Score on test data (Decision Tree):", r2)


