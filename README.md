# E-commerce_Analysis   
This is my final project of data analysis as a practice for the purpose of learning.     
# E-commerce Return Rate Reduction Analysis   
Inside this project I dwelve into the deep knowledge of Data Analysis and a part of Machine Learning Model.   
The following are the core parts that are used during the projects:-   
  a.) Data Analysis ans Visualization.   
  b.) Machine learning Model.   
  c.) Sql part   

Working on this project I get to know and learn about a lots of technologies and many things.   
Lets describe this project:-------------   
firstly i had took up a "E-commerce sales database". then i had analyse that database , the basic task is to analyse the return rate of products by the customers. The main objective is to Identify why customers return products and how return rates vary by category, geography, and marketing channel.This  project is completed into some phases:-   

Database used :- E-commerce returns synthetic data.csv   
Tools used:- Python, PowerBi,  Mysql, vscode.   

"eda_analysis.py file":- contains all script of data analysis and machine model   

1.) loading and cleaning data:- In this phase i had load the dataset into my vscode  for purpose of cleaning ,analysis, exploring some fields.   
2.) Analyze Return % by Category, Location, Channel: - Return rate by category,Return Rate by Location,Return Rate by payment channel .     
3.) Predict Returns using Logistic Regression: Build a logistic regression model to predict whether a product will be returned based on some features.In this phase i had train and test the model that will gives the   return probability of any order.   
4.) Export high risk products: Then,i had export the high risk product into the seperate csv file through the practically use of model .here we set the threshhold-returns predicted with>50 % probability.      
5.) Export to csv "high_risk_product.csv":- all the products that have returned rate greater than 50% are predicted from original file and are placed in this seperate file.   
6.) Load into MySQL for Power BI:- now load the "high_risk_product" file  data into mysql for creating a new fresh database and tables   

"export_scripy.py file" :- This file contain a python script that will connect the mysql to vscode and we automatically export the desired file.   
"ecommerce_risky"  :- This is database created in mysql and table "high_risk_products" is imported inside this  

7.) load the data from mysql to PowerBi:- after loading required data to powerBi ,then  start making Visuals for it as showing in screenshots and powerbi file.   



