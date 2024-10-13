import pandas as pd
import datetime as dt

#! an extremely detailed analysis, containing lots of reusable and effective methods.

# Load datasets
customers = pd.read_csv("2.0customers.csv")
orders = pd.read_csv("2.1orders.csv")
products = pd.read_csv("2.2products.csv")
ratings = pd.read_csv("2.3ratings.csv")

# convert "order_date" column to datetime format instead of string
orders['order_date'] = pd.to_datetime(orders['order_date'])

# merge datasets
merged_data = orders.merge(products, on='product_id').merge(customers, on='customer_id')
print("********************* merged_data")
print(merged_data)

#! calculate revenue and units sold for each product
product_performance = merged_data.groupby('product_name').agg({
    'price': 'sum', # will give the revenue
    'quantity': 'sum' # will give the total units sold
}).reset_index()

# sort by revenue and units sold
top_products_revenue = product_performance.sort_values(by="price", ascending=False)
top_products_units = product_performance.sort_values(by="quantity", ascending=False)
print("********************* top_products_revenue")
print(top_products_revenue)
print("********************* top_products_units")
print(top_products_units)

#! identify top clients for the last month

# filter orders for the last month
last_month = merged_data[merged_data['order_date'].dt.month == merged_data['order_date'].dt.month.max()]

# calculate total purchase amount for each customer in the last month

top_clients_last_month = last_month.groupby('name').agg({
    'price':'sum'
}).reset_index()

# sort to get top clients
top_clients_last_month = top_clients_last_month.sort_values(by='price', ascending=False)
print("********************* top_clients_last_month")
print(top_clients_last_month)

#! RFM Analysis (Recency, Frequency, Monetary)

# Calculate the maximum order date to use as a reference
latest_date = merged_data['order_date'].max() + dt.timedelta(days=1)

# Calculate RFM values
rfm = merged_data.groupby('customer_id').agg({
    'order_date': lambda x: (latest_date - x.max()).days, # Recency: days since last order
    'order_id': 'count',  # Frequency: number of orders
    'price': 'sum'  # Monetary: total money spent
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'price': 'monetary'
})

# Assign scores from 1 to 5 for each metric
rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]) # Higher score for lower recency
rfm['frequency_score'] = pd.qcut(rfm['frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine scores
rfm['rfm_score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)

# Display top customers
rfm_sorted = rfm.sort_values(by=['rfm_score'], ascending=False)

print("********************* rfm_sorted")
print(rfm_sorted)

#! top reviewed products and average number of reviews per custommer

top_reviewed_products = ratings.groupby('product_id').size().reset_index(name='num_reviews')
top_reviewed_products = top_reviewed_products.merge(products, on='product_id')  # Add product names for better clarity
top_reviewed_products = top_reviewed_products.sort_values(by='num_reviews', ascending=False)

reviews_per_customer = ratings.groupby('customer_id').size()
avg_reviews_per_customer = reviews_per_customer.mean()

# Group by product_id, count reviews and compute average rating
product_reviews = ratings.groupby('product_id').agg({
    'rating': ['count', 'mean']
}).reset_index()

# Flatten the column headers for easier reference
product_reviews.columns = ['product_id', 'num_reviews', 'avg_rating']

product_reviews = product_reviews.merge(products, on='product_id')
top_reviewed_with_avg_rating = product_reviews.sort_values(by='avg_rating', ascending=False)

print("*********************")
print(top_reviewed_with_avg_rating.head())

#! editing customer table as it would include "total amount they spent", "last date of purchase", and "number of reviews"

total_spent = merged_data.groupby('customer_id').apply(lambda x: (x['quantity'] * x['price']).sum()).reset_index(name='total_spent')
last_purchase = merged_data.groupby('customer_id')['order_date'].max().reset_index(name='last_purchase_date')
num_reviews = ratings.groupby('customer_id').size().reset_index(name='num_reviews')

# Merge these aggregations back to the customers DataFrame
customers_updated = customers.merge(total_spent, on='customer_id', how='left')\
                             .merge(last_purchase, on='customer_id', how='left')\
                             .merge(num_reviews, on='customer_id', how='left')

# Fill NaN values with appropriate defaults
customers_updated['total_spent'] = customers_updated['total_spent'].fillna(0)
customers_updated['num_reviews'] = customers_updated['num_reviews'].fillna(0)

print("*********************")
print(customers_updated.head())