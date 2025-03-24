# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#  Load the dataset
file_path = r"C:\Users\KIIT\Desktop\AD lab\REPORT 24th march\project root folder\Real estate.csv"  # Replace with the path to your dataset
data = pd.read_csv('Real estate.csv')

#  Use only the first 1000 rows
data_1000 = data.iloc[:1000]

# Select relevant features and target
X = data_1000[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
y = data_1000['Y house price of unit area']

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Save the trained model to a .pkl file
model_file = 'house_price_model_1000.pkl'
joblib.dump(model, model_file)

#  Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics and saved model path
print(f'Model saved as: {model_file}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
