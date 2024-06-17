# Customer Segmentation using K-Means Clustering

## Description
This project demonstrates how to segment customers based on their buying behavior using the K-Means clustering algorithm in Python. It guides you step-by-step from preparing the data to clustering it.

## Acknowledgements
Dataset sourced from the [UCI ML Repository: Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail).

## Objective
1. Clean the dataset.
2. Build a clustering model to segment customers.
3. Fine-tune the model and compare metrics.

## Steps

### Data Cleaning
1. **Handle Missing Values**: Identify and manage missing data.
2. **Create Attributes**:
   - **Monetary**: Total amount spent by each customer.
   - **Frequency**: Number of purchases by each customer.
   - **Recency**: Days since the last purchase.
3. **Merge Data**: Combine necessary datasets.
4. **Outlier Analysis**: Identify and manage outliers in the data.

### Model Building
1. **K-Means Clustering**: Apply the K-Means algorithm.
2. **Elbow Curve**: Determine the optimal number of clusters.
3. **Visualization**: Use boxplots to visualize clusters.

## Usage
1. **Data Cleaning**: Follow the steps in the notebook to clean the data.
2. **Model Training**: Train the K-Means model and tune parameters.
3. **Evaluation**: Visualize and evaluate the clusters.

## Conclusion
This project helps you understand customer segmentation using K-Means clustering, providing insights into customer behavior to enhance marketing strategies.
