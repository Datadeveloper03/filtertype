import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define file paths
data_path = "mining_emissions_data.csv"
models_dir = "models"

# Ensure the 'models' directory exists
os.makedirs(models_dir, exist_ok=True)

# Load dataset
print("📂 Loading dataset...")
data = pd.read_csv(data_path)

# Display first few rows
print("\n🧐 First 5 rows of the dataset:")
print(data.head())

# Check missing values
print("\n📊 Checking for missing values:")
print(data.isnull().sum())

# Check column types
print("\n🔍 Column Data Types:")
print(data.dtypes)

# Encode categorical target variable (Filter_Type)
print("\n🔄 Encoding categorical variables...")
label_encoder = LabelEncoder()
data['Filter_Type'] = label_encoder.fit_transform(data['Filter_Type'])

# Select features & target
X = data.drop(columns=['Filter_Type'])
y = data['Filter_Type']

# Normalize numerical features
print("📏 Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training & testing sets
print("\n✂️ Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
print("\n🤖 Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete! Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Save model & preprocessing tools
print("\n💾 Saving trained model and preprocessing tools...")
joblib.dump(model, os.path.join(models_dir, "filter_recommendation_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))

print("\n🎉 All files saved in the 'models/' folder. Training process completed successfully!")
