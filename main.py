import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")

df['Discount'] =   df['Base Price'] - df['Total Price']
df['Discount'] = np.round(df['Discount'] , 2)

df['Discount percentage'] = 100 * df['Discount'] / df['Base Price']
df['Discount percentage'] = np.round(df['Discount percentage'] , 2)
df = df.rename(columns={'Store ID': 'StoreID'})
df = df.rename(columns={'Total Price': 'Total_Price'})
df = df.rename(columns={'Base Price': 'Base_Price'})
df = df.rename(columns={'Units Sold': 'Units_Sold'})
df = df.rename(columns={'Discount percentage': 'Discount_percentage'})

def classify_demand(row):
    if row['Discount'] > 20 and row['Units_Sold'] > 20:
        return "High"
    elif row['Discount'] > 10 and (row['Units_Sold'] >= 10 and row['Units_Sold'] <= 20):
        return "Med"
    else:
        return "Low"

df['Demand'] = df.apply(classify_demand, axis=1)

print(df.head(20))