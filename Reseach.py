# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Defining the dataset
data = pd.DataFrame({
    'gender': ['female', 'male', 'female', 'male', 'female', 'male', 'female'],
    'race/ethnicity': ['group D', 'group D', 'group D', 'group B', 'group D', 'group C', 'group E'],
    'parental level of education': ['some college', "associate's degree", 'some college', 'some college', 
                                    "associate's degree", 'some high school', "associate's degree"],
    'lunch': ['standard', 'standard', 'free/reduced', 'free/reduced', 'standard', 'standard', 'standard'],
    'test preparation course': ['completed', 'none', 'none', 'none', 'none', 'none', 'none'],
    'math score': [59, 96, 57, 70, 83, 68, 82],
    'reading score': [70, 93, 76, 70, 85, 57, 83],
    'writing score': [78, 87, 77, 63, 86, 54, 80]
})

# Defining features and target
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score', 'writing score']
target = 'math score'

X = data[features]
y = data[target]

# Preprocessing: One-hot encoding for categorical variables
categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
numeric_features = ['reading score', 'writing score']

# Define the column transformer to handle the one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
