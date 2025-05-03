import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
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
    # Define features
    numeric_features = ['tenure', 'monthly_charges']
    categorical_features = ['contract', 'payment_method']
    
    # Validate features exist
    missing_features = [col for col in numeric_features + categorical_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Handle missing values
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor

# Model training with ensemble and stacking
def train_model(X, y, preprocessor):
    """Train ensemble and stacked models."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ]
    
    # Define stacking classifier
    stacked_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stacked_model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print("Cross-Validation ROC AUC Scores:", cv_scores.mean())
    
    return pipeline, X_test, y_test

# Save model securely
def save_model(model, filename):
    """Save model to disk securely."""
    if not filename.endswith('.pkl'):
        raise ValueError("Model file must have .pkl extension.")
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

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
        preprocessor = preprocess_data(df)
        
        # Train and evaluate
        model, X_test, y_test = train_model(X, y, preprocessor)
        
        # Save model
        save_model(model, "churn_model_v1.pkl")
        
        # Deployment notes
        deployment_notes()
        
    except Exception as e:
        print(f"Error: {str(e)}")
