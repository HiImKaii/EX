import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class ForestFirePredictionModel:
    """
    Class for training and evaluating forest fire prediction models.
    Implements both Bayesian Network and Multivariate Logistic Regression models.
    """
    
    def __init__(self, data_path, test_size=0.3, random_state=42, scaler_type='standard'):
        """
        Initialize the model with data and parameters.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the extracted features
        test_size : float
            Proportion of data to use for testing (default: 0.3)
        random_state : int
            Random seed for reproducibility (default: 42)
        scaler_type : str
            Type of feature scaling ('standard' or 'minmax')
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_type = scaler_type
        
        # Initialize model containers
        self.bn_model = None
        self.lr_model = None
        self.feature_scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        
        # Load and preprocess data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the dataset"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Check if 'fire' column exists
        if 'fire' not in self.data.columns:
            raise ValueError("Dataset must contain a 'fire' column!")
        
        print(f"Dataset loaded with {self.data.shape[0]} samples and {self.data.shape[1]} features.")
        print(f"Fire positive samples: {self.data['fire'].sum()} ({self.data['fire'].mean()*100:.2f}%)")
        
        # Basic preprocessing
        # Drop any rows with NaN values
        self.data = self.data.dropna()
        print(f"After dropping NaN values: {self.data.shape[0]} samples")
        
        # Extract features and target
        self.X = self.data.drop('fire', axis=1)
        self.y = self.data['fire']
        
        # Save feature names
        self.features = self.X.columns.tolist()
        print(f"Features: {self.features}")
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        
        # Scale features
        if self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        self.X_train_scaled = self.feature_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        print(f"Data split into {self.X_train.shape[0]} training samples and {self.X_test.shape[0]} testing samples")
    
    def train_bayesian_network(self, scoring_method='bic', max_indegree=4, discretize=True, bins=5):
        """
        Train a Bayesian Network model for forest fire prediction.
        
        Parameters:
        -----------
        scoring_method : str
            Scoring method for structure learning ('bic' or 'k2')
        max_indegree : int
            Maximum number of parents per node
        discretize : bool
            Whether to discretize continuous variables
        bins : int
            Number of bins for discretization
        """
        print("\n=== Training Bayesian Network ===")
        
        # For Bayesian Network, we need discretized data
        if discretize:
            print(f"Discretizing continuous variables into {bins} bins...")
            # Make a copy of the data to discretize
            data_discrete = self.data.copy()
            
            # Discretize all numerical columns except the target
            for col in data_discrete.columns:
                if col != 'fire' and pd.api.types.is_numeric_dtype(data_discrete[col]):
                    data_discrete[col] = pd.qcut(data_discrete[col], q=bins, labels=False, duplicates='drop')
            
            # Convert all columns to string categories for pgmpy
            data_discrete = data_discrete.astype(str)
        else:
            data_discrete = self.data.copy()
        
        # Structure learning
        print("Learning Bayesian Network structure...")
        hc = HillClimbSearch(data_discrete)
        best_model = hc.estimate(
            scoring_method=scoring_method,
            max_indegree=max_indegree,
            return_independencies=True
        )
        
        print(f"Learned structure: {best_model.edges()}")
        
        # Create Bayesian Network model
        self.bn_model = BayesianNetwork(best_model.edges())
        
        # Parameter learning
        print("Estimating parameters...")
        self.bn_model.fit(data_discrete, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)
        
        # Create inference object
        self.bn_inference = VariableElimination(self.bn_model)
        
        print("Bayesian Network training complete!")
        return self.bn_model
    
    def train_logistic_regression(self, solver='liblinear', penalty='l2', C=1.0, max_iter=1000, cv=5):
        """
        Train a Multivariate Logistic Regression model for forest fire prediction.
        
        Parameters:
        -----------
        solver : str
            Algorithm for optimization problem
        penalty : str
            Regularization type ('l1', 'l2', 'elasticnet', or 'none')
        C : float
            Inverse of regularization strength
        max_iter : int
            Maximum number of iterations for solver
        cv : int
            Number of folds for cross-validation
        """
        print("\n=== Training Multivariate Logistic Regression ===")
        
        # Define parameter grid for grid search
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'] if solver == 'liblinear' else ['l2'],
            'solver': [solver]
        }
        
        # Initialize model
        base_lr = LogisticRegression(max_iter=max_iter, random_state=self.random_state)
        
        # Grid search with cross-validation
        print(f"Performing grid search with {cv}-fold cross-validation...")
        grid_search = GridSearchCV(
            base_lr, 
            param_grid, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        self.lr_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Print feature importance
        coef = self.lr_model.coef_[0]
        importance = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': coef,
            'Absolute': np.abs(coef)
        })
        importance = importance.sort_values('Absolute', ascending=False)
        print("\nFeature importance:")
        print(importance[['Feature', 'Coefficient']].head(10))
        
        print("Logistic Regression training complete!")
        return self.lr_model
    
    def evaluate_models(self, threshold=0.5):
        """
        Evaluate both trained models on the test set.
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for binary classification
        """
        print("\n=== Model Evaluation ===")
        
        results = {}
        
        # Evaluate Logistic Regression
        if self.lr_model is not None:
            print("\nLogistic Regression Model:")
            y_prob_lr = self.lr_model.predict_proba(self.X_test_scaled)[:, 1]
            y_pred_lr = (y_prob_lr >= threshold).astype(int)
            
            # Calculate metrics
            acc_lr = accuracy_score(self.y_test, y_pred_lr)
            prec_lr = precision_score(self.y_test, y_pred_lr)
            rec_lr = recall_score(self.y_test, y_pred_lr)
            f1_lr = f1_score(self.y_test, y_pred_lr)
            auc_lr = roc_auc_score(self.y_test, y_prob_lr)
            
            print(f"Accuracy: {acc_lr:.4f}")
            print(f"Precision: {prec_lr:.4f}")
            print(f"Recall: {rec_lr:.4f}")
            print(f"F1 Score: {f1_lr:.4f}")
            print(f"ROC AUC: {auc_lr:.4f}")
            
            # Confusion matrix
            cm_lr = confusion_matrix(self.y_test, y_pred_lr)
            print("\nConfusion Matrix:")
            print(cm_lr)
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred_lr))
            
            # Store results
            results['lr'] = {
                'accuracy': acc_lr,
                'precision': prec_lr,
                'recall': rec_lr,
                'f1_score': f1_lr,
                'roc_auc': auc_lr,
                'y_prob': y_prob_lr,
                'y_pred': y_pred_lr,
                'confusion_matrix': cm_lr
            }
        
        # Evaluate Bayesian Network
        if self.bn_model is not None:
            print("\nBayesian Network Model:")
            
            # Bayesian Network prediction is more complex as we need to do inference
            # We'll create a function to predict for each test sample
            
            y_pred_bn = []
            y_prob_bn = []
            
            # Discretize the test data the same way as training data
            X_test_discrete = self.X_test.copy()
            for col in X_test_discrete.columns:
                if pd.api.types.is_numeric_dtype(X_test_discrete[col]):
                    X_test_discrete[col] = pd.qcut(self.X_test[col], q=5, labels=False, duplicates='drop')
            
            X_test_discrete = X_test_discrete.astype(str)
            
            print("Running inference on test data...")
            for i in range(len(X_test_discrete)):
                evidence = X_test_discrete.iloc[i].to_dict()
                try:
                    query_result = self.bn_inference.query(variables=['fire'], evidence=evidence)
                    prob = float(query_result.values[1])  # Probability of fire=1
                    y_prob_bn.append(prob)
                    y_pred_bn.append(1 if prob >= threshold else 0)
                except Exception as e:
                    print(f"Error in inference for sample {i}: {e}")
                    # Use most common class as fallback
                    y_prob_bn.append(0.5)
                    y_pred_bn.append(self.y_train.mode()[0])
            
            # Calculate metrics
            acc_bn = accuracy_score(self.y_test, y_pred_bn)
            prec_bn = precision_score(self.y_test, y_pred_bn)
            rec_bn = recall_score(self.y_test, y_pred_bn)
            f1_bn = f1_score(self.y_test, y_pred_bn)
            auc_bn = roc_auc_score(self.y_test, y_prob_bn)
            
            print(f"Accuracy: {acc_bn:.4f}")
            print(f"Precision: {prec_bn:.4f}")
            print(f"Recall: {rec_bn:.4f}")
            print(f"F1 Score: {f1_bn:.4f}")
            print(f"ROC AUC: {auc_bn:.4f}")
            
            # Confusion matrix
            cm_bn = confusion_matrix(self.y_test, y_pred_bn)
            print("\nConfusion Matrix:")
            print(cm_bn)
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred_bn))
            
            # Store results
            results['bn'] = {
                'accuracy': acc_bn,
                'precision': prec_bn,
                'recall': rec_bn,
                'f1_score': f1_bn,
                'roc_auc': auc_bn,
                'y_prob': y_prob_bn,
                'y_pred': y_pred_bn,
                'confusion_matrix': cm_bn
            }
        
        self.results = results
        return results
    
    def plot_roc_curves(self):
        """Plot ROC curves for both models"""
        if not hasattr(self, 'results'):
            print("Run evaluate_models() first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        # ROC curve for Logistic Regression
        if 'lr' in self.results:
            fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.results['lr']['y_prob'])
            plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {self.results['lr']['roc_auc']:.4f})")
        
        # ROC curve for Bayesian Network
        if 'bn' in self.results:
            fpr_bn, tpr_bn, _ = roc_curve(self.y_test, self.results['bn']['y_prob'])
            plt.plot(fpr_bn, tpr_bn, label=f"Bayesian Network (AUC = {self.results['bn']['roc_auc']:.4f})")
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300)
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance for the Logistic Regression model"""
        if self.lr_model is None:
            print("Train the Logistic Regression model first.")
            return
        
        # Extract coefficients
        coef = self.lr_model.coef_[0]
        
        # Create DataFrame for plotting
        importance = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': coef,
            'Absolute': np.abs(coef)
        })
        importance = importance.sort_values('Absolute', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
        plt.title('Logistic Regression Feature Importance')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()
    
    def plot_bayesian_network(self):
        """Plot the Bayesian Network structure"""
        if self.bn_model is None:
            print("Train the Bayesian Network model first.")
            return
        
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            import networkx as nx
            
            # Get networkx graph
            G = nx.DiGraph()
            G.add_edges_from(self.bn_model.edges())
            
            # Plot
            plt.figure(figsize=(12, 10))
            pos = graphviz_layout(G, prog='dot') if graphviz_layout else nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
                          font_size=10, arrows=True, arrowsize=20, font_weight='bold')
            plt.title('Bayesian Network Structure')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('bayesian_network.png', dpi=300)
            plt.show()
        except ImportError:
            print("Graphviz or networkx not available. Cannot plot Bayesian Network.")
    
    def save_models(self, output_dir='models'):
        """
        Save trained models to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        """
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save Logistic Regression model
        if self.lr_model is not None:
            lr_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
            with open(lr_path, 'wb') as f:
                pickle.dump({
                    'model': self.lr_model,
                    'scaler': self.feature_scaler,
                    'features': self.features
                }, f)
            print(f"Logistic Regression model saved to {lr_path}")
        
        # Save Bayesian Network model
        if self.bn_model is not None:
            bn_path = os.path.join(output_dir, 'bayesian_network_model.pkl')
            with open(bn_path, 'wb') as f:
                pickle.dump({
                    'model': self.bn_model,
                    'inference': self.bn_inference,
                    'features': self.features
                }, f)
            print(f"Bayesian Network model saved to {bn_path}")
    
    @staticmethod
    def load_model(model_path):
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        
        Returns:
        --------
        dict
            Dictionary containing the model and associated objects
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

def main():
    """
    Main function to demonstrate model training and evaluation workflow
    """
    # File path - modify this to the actual path of your dataset
    data_path = "balanced_fire_data.csv"
    
    # Initialize model
    model = ForestFirePredictionModel(data_path, test_size=0.3, random_state=42)
    
    # Train Logistic Regression model
    model.train_logistic_regression(solver='liblinear', cv=5)
    
    # Train Bayesian Network model
    model.train_bayesian_network(max_indegree=3, bins=5)
    
    # Evaluate models
    model.evaluate_models()
    
    # Plot results
    model.plot_roc_curves()
    model.plot_feature_importance()
    model.plot_bayesian_network()
    
    # Save models
    model.save_models(output_dir='models')
    
    print("Forest fire modeling complete!")

if __name__ == "__main__":
    main()
