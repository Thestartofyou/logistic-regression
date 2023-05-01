import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
data = pd.read_csv('home_sales.csv')

# Preprocess data
# Here you would perform any necessary data cleaning and feature engineering, such as removing missing values, encoding categorical variables, and scaling numeric features.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('sold_within_20_days', axis=1), data['sold_within_20_days'], test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))

# Use model to make predictions
new_data = pd.DataFrame({'area': [5000], 'num_bedrooms': [3], 'num_bathrooms': [2], 'age': [10], 'size': [2000], 'garage': [1]})
prob = model.predict_proba(new_data)[:,1]
print('Probability of home being sold within 20 days:', prob)
