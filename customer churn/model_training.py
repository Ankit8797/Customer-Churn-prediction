import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_data, preprocess_data

def train_model():
    # Load data
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocess and get encoders
    X, y, encoders = preprocess_data(df, fit_encoders=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print("Model and encoders saved successfully.")
    return model, encoders, X_test, y_test

if __name__ == '__main__':
    train_model()