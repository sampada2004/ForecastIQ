import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score , accuracy_score
from sklearn import tree

import warnings
import pickle

global data

warnings.filterwarnings("ignore")

def load_process_data():
    #Loading Dataset
    data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")


    #Feature Engineering

    data['Discount'] =   data['Base Price'] - data['Total Price']
    data['Discount'] = np.round(data['Discount'] , 2)

    data['Discount percentage'] = 100 * data['Discount'] / data['Base Price']
    data['Discount percentage'] = np.round(data['Discount percentage'] , 2)
    data['Demand'] = data.apply(classify_demand, axis=1)
    


    data=data[(data['Discount']>=0)]
    print()

    # Handle missing values if any
    data.fillna(method='ffill', inplace=True)

    # Encode categorical target variable
    le = LabelEncoder()
    data['Demand'] = le.fit_transform(data['Demand'])

    return data


def classify_demand(row):
    if row['Units Sold'] >= 50:
        return "High"
    elif row['Discount'] > 20 and row['Units Sold'] > 20:
        return "High"
    elif row['Discount'] > 10 and (row['Units Sold'] >= 10 and row['Units Sold'] <= 20):
        return "Med"
    elif row['Units Sold'] > 20 and row['Units Sold'] < 50:
        return "Med"
    elif row['Discount']<10 and row['Units Sold']<10:
        return "Low"
    else:
        return "High"


            
def split_train_data(data):

    #Data Splitting
    
    x = data.drop(columns=['Demand'])
    y = data['Demand']

    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.7 ,random_state=42)

    #Initialize Decision Tree Regression

    model = DecisionTreeRegressor(random_state = 42 , ccp_alpha=0.02)

    #Train the model

    model.fit(x_train , y_train)

    #Dump into pickle file

   # pickle.dump(model ,open('Content.pkl' , 'wb'))
    #model = pickle.load(open('Content.pkl','rb'))

    #ccp_alpha 
    #path = model.cost_complexity_pruning_path(x_train, y_train)
    #ccp_alphas, impurities = path.ccp_alphas, path.impurities

    #Predict on Test

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)


    #Metrics

    mse = mean_squared_error(y_test , y_pred)
    r2 = r2_score(y_test , y_pred)
    acc = accuracy_score(y_test , y_pred)
    train_accuracy = accuracy_score(y_train, y_pred_train)


    print(f"Mean Squared Error : {mse}")
    print(f"R^2 Error : {r2}")
    print(f"Test Accuracy : {acc}")
    print(f"Train Accuracy: {train_accuracy}")




    return model

    

def input_features():
     
    #User Input
    base_price = float(input("Enter Base Price: "))
    total_price = float(input("Enter Total Price: "))
    units_sold = int(input("Enter Units Sold: "))


    
    discount = base_price - total_price
    discount_percentage = 100 * discount / base_price
    
    return pd.DataFrame({
        'ID' : [0],
        'Store ID': [0] ,
        'Total Price': [total_price],
        'Base Price': [base_price],
        'Units Sold': [units_sold],
        'Discount': [np.round(discount, 2)],
        'Discount percentage': [np.round(discount_percentage, 2)] 
       
        })


if __name__ == "__main__":
    data = load_process_data()
    model = split_train_data(data)
    
    new_data = input_features()
    prediction = model.predict(new_data)
    
    # Print the prediction
    if prediction == 2:
        print("Demand for the product is High")
    elif prediction == 1:
        print("Demand for the Product is Average")
    elif prediction == 0:
        print("Demand for the product is Low")

    # Visualize the tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=data.drop(columns=['Demand']).columns, filled=True, fontsize=10)
    plt.show()