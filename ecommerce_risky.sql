create database Ecommerce_risky;
use Ecommerce_risky;
create table high_risk_products (
order_ID varchar(50),
product_ID varchar(50),
product_category varchar(100),
product_price float,
user_ID varchar(50),
user_location varchar (100),
user_age int,
user_gender varchar(100),
payment_method varchar(50),
shipping_method varchar(50),
discount_Applied FLOAT,
returned_Flag INT,
return_Prob FLOAT
);
SELECT * FROM high_risk_products
limit 10;
show tables;



