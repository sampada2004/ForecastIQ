import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree


global data

def load_process_data():
    #Loading Dataset
    data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")


#Feature Engineering
    data['Base Price']=int(input("enter Base Price"))
    data['Total Price']=int(input("Enter Total Price"))
    data['Discount'] =   data['Base Price'] - data['Total Price']
    data['Discount'] = np.round(data['Discount'] , 2)

    data['Discount percentage'] = 100 * data['Discount'] / data['Base Price']
    data['Discount percentage'] = np.round(data['Discount percentage'] , 2)


    def classify_demand(row):
        if row['Discount'] > 20 and row['Units Sold'] > 20  :
            return "High"
        elif row['Discount'] > 10 and (row['Units Sold'] >= 10 and row['Units Sold'] <= 20):
            return "Med"
        elif row['Units Sold'] >= 50:
            return "High"
        elif row['Units Sold'] > 20 and row['Units Sold'] <50:
            return "Med"
        else:
            return "Low"

    data['Demand'] = data.apply(classify_demand, axis=1)


    data=data[(data['Discount']>=0)]
    print()

    # Handle missing values if any
    data.fillna(method='ffill', inplace=True)

    # Encode categorical target variable
    le = LabelEncoder()
    data['Demand'] = le.fit_transform(data['Demand'])

    return data

def split_train_data(data):

    #Data Splitting

    x = data.drop(columns=['Demand'])
    y = data['Demand']

    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.8 ,random_state=42)

    #Initialize Decission Tree Regression

    model = DecisionTreeRegressor(random_state = 42)

    #Train the model

    model.fit(x_train , y_train)

    #Predict on Test

    y_pred = model.predict(x_test)

    #Metrics

    mse = mean_squared_error(y_test , y_pred)
    r2 = r2_score(y_test , y_pred)

    print(f"Mean Squared Error : {mse}")
    print(f"R^2 Error : {r2}")

    tree.plot_tree(model)
    plt.show()

if __name__ == "__main__":
    data = load_process_data()
    split_train_data(data)