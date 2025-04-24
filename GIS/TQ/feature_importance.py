import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import os
import pickle

def load_data(data_path):
    """
    Load and preprocess dataset for feature importance analysis
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the extracted features
    
    Returns:
    --------
    X : DataFrame
        Feature data
    y : Series
        Target variable
    feature_names : list
        List of feature names
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Check if 'fire' column exists
    if 'fire' not in data.columns:
        raise ValueError("Dataset must contain a 'fire' column!")
    
    # Basic preprocessing
    data = data.dropna()
    
    # Extract features and target
    X = data.drop('fire', axis=1)
    y = data['fire']
    
    print(f"Dataset loaded with {data.shape[0]} samples and {data.shape[1]} features.")
    print(f"Fire positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    
    return X, y, X.columns.tolist()

def evaluate_logistic_regression_importance(X, y, random_state=42):
    """
    Evaluate feature importance using Logistic Regression coefficients
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target variable
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame
        Feature importance scores
    """
    print("\nEvaluating feature importance using Logistic Regression...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Get feature importance (coefficients)
    coef = lr.coef_[0]
    
    # Evaluate model
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Logistic Regression ROC AUC: {auc:.4f}")
    
    # Create DataFrame for importance scores
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coef,
        'Absolute': np.abs(coef)
    })
    importance_df = importance_df.sort_values('Absolute', ascending=False)
    
    return importance_df

def evaluate_random_forest_importance(X, y, random_state=42):
    """
    Evaluate feature importance using Random Forest feature importance
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target variable
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame
        Feature importance scores
    """
    print("\nEvaluating feature importance using Random Forest...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Evaluate model
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Random Forest ROC AUC: {auc:.4f}")
    
    # Create DataFrame for importance scores
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def evaluate_permutation_importance(X, y, random_state=42):
    """
    Evaluate feature importance using permutation importance
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Target variable
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame
        Feature importance scores
    """
    print("\nEvaluating feature importance using permutation importance...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (as base model)
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )
    
    # Create DataFrame for importance scores
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def plot_feature_importance(importance_df, method_name, output_file=None):
    """
    Plot feature importance scores
    
    Parameters:
    -----------
    importance_df : DataFrame
        Feature importance scores
    method_name : str
        Name of the method used for feature importance
    output_file : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, len(importance_df) * 0.4))
    
    if 'Coefficient' in importance_df.columns:
        # For logistic regression
        sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette='viridis')
        plt.axvline(x=0, color='r', linestyle='--')
    else:
        # For other methods
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    
    plt.title(f'Feature Importance - {method_name}')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def combined_importance_analysis(lr_importance, rf_importance, perm_importance, output_file=None):
    """
    Create combined feature importance analysis
    
    Parameters:
    -----------
    lr_importance : DataFrame
        Logistic Regression importance scores
    rf_importance : DataFrame
        Random Forest importance scores
    perm_importance : DataFrame
        Permutation importance scores
    output_file : str, optional
        Path to save the plot
    """
    # Normalize importance scores
    lr_norm = lr_importance.copy()
    lr_norm['Normalized'] = lr_norm['Absolute'] / lr_norm['Absolute'].max()
    
    rf_norm = rf_importance.copy()
    rf_norm['Normalized'] = rf_norm['Importance'] / rf_norm['Importance'].max()
    
    perm_norm = perm_importance.copy()
    perm_norm['Normalized'] = perm_norm['Importance'] / perm_norm['Importance'].max()
    
    # Get all features
    all_features = list(set(lr_norm['Feature']) | set(rf_norm['Feature']) | set(perm_norm['Feature']))
    
    # Create combined DataFrame
    combined = pd.DataFrame({'Feature': all_features})
    
    # Add normalized scores from each method
    combined = combined.merge(
        lr_norm[['Feature', 'Normalized']], on='Feature', how='left'
    ).rename(columns={'Normalized': 'LogReg'})
    
    combined = combined.merge(
        rf_norm[['Feature', 'Normalized']], on='Feature', how='left'
    ).rename(columns={'Normalized': 'RandForest'})
    
    combined = combined.merge(
        perm_norm[['Feature', 'Normalized']], on='Feature', how='left'
    ).rename(columns={'Normalized': 'Permutation'})
    
    # Fill NaN values with 0
    combined = combined.fillna(0)
    
    # Calculate average score
    combined['Average'] = combined[['LogReg', 'RandForest', 'Permutation']].mean(axis=1)
    
    # Sort by average score
    combined = combined.sort_values('Average', ascending=False)
    
    # Plot
    plt.figure(figsize=(14, len(combined) * 0.5))
    
    # Melt DataFrame for easier plotting
    melted = pd.melt(
        combined, 
        id_vars=['Feature'], 
        value_vars=['LogReg', 'RandForest', 'Permutation', 'Average'],
        var_name='Method', value_name='Importance'
    )
    
    # Plot
    sns.barplot(x='Importance', y='Feature', hue='Method', data=melted, palette='viridis')
    plt.title('Combined Feature Importance Analysis')
    plt.legend(title='Method')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return combined

def main():
    """
    Main function to evaluate feature importance for forest fire prediction
    """
    # File path - modify to the actual path of your dataset
    data_path = "balanced_fire_data.csv"
    
    # Load data
    X, y, feature_names = load_data(data_path)
    
    # Create output directory
    output_dir = "feature_importance"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Evaluate feature importance using Logistic Regression
    lr_importance = evaluate_logistic_regression_importance(X, y)
    plot_feature_importance(
        lr_importance, 
        "Logistic Regression", 
        os.path.join(output_dir, "lr_importance.png")
    )
    
    # Evaluate feature importance using Random Forest
    rf_importance = evaluate_random_forest_importance(X, y)
    plot_feature_importance(
        rf_importance, 
        "Random Forest", 
        os.path.join(output_dir, "rf_importance.png")
    )
    
    # Evaluate feature importance using permutation importance
    perm_importance = evaluate_permutation_importance(X, y)
    plot_feature_importance(
        perm_importance, 
        "Permutation Importance", 
        os.path.join(output_dir, "perm_importance.png")
    )
    
    # Combined analysis
    combined = combined_importance_analysis(
        lr_importance,
        rf_importance,
        perm_importance,
        os.path.join(output_dir, "combined_importance.png")
    )
    
    # Save importance scores to CSV
    lr_importance.to_csv(os.path.join(output_dir, "lr_importance.csv"), index=False)
    rf_importance.to_csv(os.path.join(output_dir, "rf_importance.csv"), index=False)
    perm_importance.to_csv(os.path.join(output_dir, "perm_importance.csv"), index=False)
    combined.to_csv(os.path.join(output_dir, "combined_importance.csv"), index=False)
    
    print("\nFeature importance analysis complete!")
    print(f"Results saved to {output_dir}/ directory")
    
    # Print top 10 features based on combined analysis
    print("\nTop 10 most important features (combined analysis):")
    for i, row in combined.head(10).iterrows():
        print(f"{i+1}. {row['Feature']} (Score: {row['Average']:.4f})")

if __name__ == "__main__":
    main() 