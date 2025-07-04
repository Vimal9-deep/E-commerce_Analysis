# Through this script, we will load the high-risk product data into a MySQL database for further analysis and reporting in Power BI

import pandas as pd
import mysql.connector #module that Connects Python with your MySQL database so we can insert data

# Load CSV
df = pd.read_csv("high_risk_products.csv")

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='host123',
    database='ecommerce_risky'
)
cursor = conn.cursor() #cursor is the object used to run SQL queries from Python.

# Insert rows
for _, row in df.iterrows():  #df.iterrows() loops over each row in your dataframe.
    cursor.execute("""
        INSERT INTO high_risk_products (
            Order_ID, Product_ID, Product_Category, Product_Price,
            User_ID, User_Location, User_Age, User_Gender,
            Payment_Method, Shipping_Method, Discount_Applied,
            Returned_Flag, Return_Prob
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit() #Confirms all changes — writes data into the actual database.
cursor.close() #Close Cursor and Connection
conn.close()
print("Data inserted into MySQL.")


#Now table contents are imported to mysql database there i.e "ecommerce_risky"  database and "high_risk_products" table

'''Python Model → high_risk_products.csv → MySQL Table → Power BI
  This is the flow of data from Python to MySQL and then to Power BI for visualization and reporting.'''
  

'''1.) we can directly import csv table high_risk_products to Power BI for creating  dashboards and visuals . it is fast and easy method
  2.) we use mysql here at mid bcz  for central data storage, 
  real world integration,
  You can join, filter, group, and optimize data before Power BI.
  Databases support access control, which CSVs do not
'''
