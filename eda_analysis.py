#step- 1,  Basic overview of file eda_analysis.py or "loading & exploring the dataset"-----------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
file_path = 'ecommerce_returns_synthetic_data.csv'
df = pd.read_csv(file_path)

# Display basic info
print("\n- First 5 rows:\n", df.head())
print("\n- Column names:\n", df.columns)
print("\n- Null values:\n", df.isnull().sum())
print("\n- Dataset shape:", df.shape)

#step 2 : cleaning the data-----------------------------------------
import pandas as pd
# Load dataset
df = pd.read_csv("ecommerce_returns_synthetic_data.csv")

# Add binary return label for modeling
df['Returned_Flag'] = df['Return_Status'].apply(lambda x: 1 if x == "Returned" else 0)

# Drop rows with missing critical values
df.dropna(subset=['Product_Category', 'User_Location', 'Payment_Method'], inplace=True)

# Show basic stats
print(df['Returned_Flag'].value_counts())
print(df[['Product_Category', 'Returned_Flag']].groupby('Product_Category').mean())


# Step 3: Analyze Return % by Category, Location, Channel------------------------------------------
# ------- Return Rate by Category -------
cat_return = df.groupby('Product_Category')['Returned_Flag'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=cat_return.index, y=cat_return.values, palette='viridis')
plt.title("Return Rate by Product Category")
plt.ylabel("Return Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------- Return Rate by Location -------
loc_return = df.groupby('User_Location')['Returned_Flag'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=loc_return.index, y=loc_return.values, palette='magma')
plt.title("Top 10 Locations by Return Rate")
plt.ylabel("Return Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------- Return Rate by Payment Channel -------
channel_return = df.groupby('Payment_Method')['Returned_Flag'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=channel_return.index, y=channel_return.values, palette='cubehelix')
plt.title("Return Rate by Payment Method")
plt.ylabel("Return Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# we will now see three bar charts:-
# Return Rate by Product Category
# Return Rate by Top 10 User Locations
# Return Rate by Payment Method (Channel)


#Step 4: Predict Returns using Logistic Regression---------------------------------------------------------------
'''explanation of Step 4
Build a logistic regression model to predict whether a product will be returned based on features like product category, price, user age, location, and more.
Regression:-It’s a classification model used to predict binary outcomes.'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Select features to use for prediction
df_model = df[['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied',
      'Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method']]
df_model = pd.get_dummies(df_model, drop_first=True)

# Labels
y = df['Returned_Flag']
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df_model, y, test_size=0.2, random_state=42)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# We will now have a modele that gives the return probability of any order


#step-5 :Export high risk products----------------------------------------------------
'''This is where your model becomes practically useful.
  Use the trained logistic regression model to predict the probability of a product being returned and export the high-risk orders (e.g., probability > 70%) into a CSV file.
Step	What we will  Do:-
predict_proba()	                     Got probabilities of return for each order
df[df['Return_Prob'] > .5]	         Filtered orders that are high risk
to_csv()	                           Saved high-risk records to CSV for reporting/dashboard'''

# Predict return probability for ALL rows (not just test set)
df_model_all = pd.get_dummies(df[[
    'Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied',
    'Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method'
]], drop_first=True)

# Align columns with model (in case some dummy variables were missing)
df_model_all = df_model_all.reindex(columns=X_train.columns, fill_value=0)

# Predict probabilities
df['Return_Prob'] = model.predict_proba(df_model_all)[:, 1]

# Set threshold — returns predicted with > 50% probability
high_risk_df = df[df['Return_Prob'] > 0.5]

# Check how many rows matched
print(f"High-Risk Products: {len(high_risk_df)} rows found.")

# Select useful columns to export (Export High-Risk Products to CSV)
columns_to_export = [
    'Order_ID', 'Product_ID', 'Product_Category', 'Product_Price',
    'User_ID', 'User_Location', 'User_Age', 'User_Gender',
    'Payment_Method', 'Shipping_Method', 'Discount_Applied',
    'Returned_Flag', 'Return_Prob'
]
# Export to CSV
high_risk_df[columns_to_export].to_csv('high_risk_products.csv', index=False)
print("Exported to high_risk_products.csv")

# Now this file is useful for reporting or dashboarding purposes at PowerBi

#step-6:  Load into MySQL for Power BI----------------------------------------
'''Load the high-risk product data into a MySQL table so it can be easily queried using SQL and connected to Power BI for dashboarding and visual analysis.'''













