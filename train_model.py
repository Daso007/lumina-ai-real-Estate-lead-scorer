import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Used to save our trained model

print("Starting model training process...")

# --- 1. Load Data ---
print("Step 1/5: Loading master training data...")
try:
    df = pd.read_csv('master_training_data.csv')
except FileNotFoundError:
    print("Error: master_training_data.csv not found.")
    print("Please run generate_data.py first.")
    exit()

# --- 2. Feature Engineering & Preprocessing ---
print("Step 2/5: Preparing data for training...")

# Define our target variable (what we want to predict)
TARGET = 'scheduled_site_visit'

# Define the features (data we use to make predictions)
# We will drop columns that are not useful for prediction (like names, emails)
features_to_drop = ['user_id', 'lead_name', 'lead_email', 'lead_phone', 'contact_date', TARGET]
features = [col for col in df.columns if col not in features_to_drop]

# Separate features (X) from the target (y)
X = df[features]
y = df[TARGET]

# Identify categorical features that need to be encoded
# The model needs all inputs to be numbers
categorical_features = ['lead_source', 'first_session_source']

# Handle missing values - a simple but effective strategy
for col in X.columns:
    if X[col].dtype == 'object':
        X[col].fillna('missing', inplace=True)
    else:
        X[col].fillna(0, inplace=True)

# Create a preprocessor to handle categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep all other (numerical) columns
)


# --- 3. Split Data into Training and Testing Sets ---
print("Step 3/5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)} records")
print(f"Testing set size: {len(X_test)} records")


# --- 4. Train the Model ---
print("Step 4/5: Training the Gradient Boosting Classifier model...")

# Create the model pipeline
# A pipeline chains the preprocessor and the model together
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train the model on the training data
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Evaluate the Model and Save ---
print("Step 5/5: Evaluating model performance...")
# Make predictions on the unseen test data
y_pred = model_pipeline.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy:.4f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Did Not Schedule', 'Scheduled Visit']))

# Save the trained model pipeline to a file
model_filename = 'lead_scorer_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"\nSuccessfully saved the trained model to {model_filename}")