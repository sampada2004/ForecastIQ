import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")

df['Discount'] =   df['Base Price'] - df['Total Price']
df['Discount'] = np.round(df['Discount'] , 2)

df['Discount percentage'] = 100 * df['Discount'] / df['Base Price']
df['Discount percentage'] = np.round(df['Discount percentage'] , 2)


print(df.head(20))
