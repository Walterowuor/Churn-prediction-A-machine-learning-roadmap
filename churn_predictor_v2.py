import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Secure data loading with validation
def load_data(file_path):
    """Load and validate dataset."""
    try:
        if not file_path.endswith('.csv'):
            raise ValueError("Only CSV files are supported.")
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

# Preprocessing with secure input handling
def preprocess_data(df):
    """Preprocess data with feature engineering and encoding."""
    numeric_features = ['tenure', 'monthly_charges']
    categorical_features = ['contract', 'payment_method']
    
    missing_features = [col for col in numeric_features + categorical_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor, numeric_features, categorical_features

# Model training with ensemble and stacking
def train_model(X, y, preprocessor):
    """Train ensemble and stacked models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ]
    
    stacked_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stacked_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print("Cross-Validation ROC AUC Scores:", cv_scores.mean())
    
    return pipeline, X_test, y_test

# Visualizations
def plot_visualizations(pipeline, X_test, y_test, numeric_features, categorical_features):
    """Generate feature importance and ROC curve plots."""
    # Feature importance
    feature_names = (numeric_features + 
                     pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features).tolist())
    importances = pipeline.named_steps['classifier'].estimators_[0][1].feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    # ROC Curve
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png")
    plt.close()

# Save model securely
def save_model(model, filename):
    """Save model to disk securely."""
    if not filename.endswith('.pkl'):
        raise ValueError("Model file must have .pkl extension.")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# Flask API for predictions
app = Flask(__name__)
load_dotenv()  # Load environment variables
model = None

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for churn prediction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate input
        required_fields = ['tenure', 'monthly_charges', 'contract', 'payment_method']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Sanitize inputs
        input_df = pd.DataFrame([data])
        for col in ['tenure', 'monthly_charges']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        if input_df.isnull().values.any():
            return jsonify({"error": "Invalid numeric inputs"}), 400
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Deployment considerations
def deployment_notes():
    """Print deployment considerations."""
    notes = """
    Deployment Considerations:
    - Use HTTPS for API endpoints to ensure secure data transmission.
    - Implement input validation to prevent injection attacks (e.g., SQL, XSS).
    - Use environment variables for sensitive configs (e.g., API keys).
    - Containerize with Docker for scalability and consistency.
    - Monitor model performance with logging and retrain periodically.
    - Secure session management with JWT or OAuth for API access.
    - Use WSGI server (e.g., Gunicorn) for production Flask deployment.
    - Enable CORS securely if API is accessed from different domains.
    """
    print(notes)

# Main execution
if __name__ == "__main__":
    try:
        # Load data (replace with your dataset path)
        df = load_data("telecom_churn.csv")
        
        # Define features and target
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Preprocess
        preprocessor, numeric_features, categorical_features = preprocess_data(df)
        
        # Train and evaluate
        model, X_test, y_test = train_model(X, y, preprocessor)
        
        # Generate visualizations
        plot_visualizations(model, X_test, y_test, numeric_features, categorical_features)
        
        # Save model
        save_model(model, "churn_model_v2.pkl")
        
        # Deployment notes
        deployment_notes()
        
        # Start Flask API (for demo; use Gunicorn in production)
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")
