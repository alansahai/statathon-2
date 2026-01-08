"""
StatFlow AI - Machine Learning Engine
Provides machine learning capabilities for survey data analysis.
Production-ready implementation matching existing engine conventions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MLEngine:
    """
    Machine Learning engine for classification, regression, and clustering.
    
    Supports:
    - Linear Regression
    - Logistic Regression
    - Random Forest (classifier + regressor)
    - KMeans Clustering
    - PCA
    
    All outputs are JSON-safe (NaN/Inf converted to None).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ML engine with a DataFrame.
        
        Args:
            df: Input DataFrame containing data
        """
        self.df = df.copy()
        self.operations_log = []
        self._log_operation("initialized", {"rows": len(df), "columns": len(df.columns)})
    
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Helper method to log operations."""
        self.operations_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            **details
        })
    
    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Return the operations log."""
        return self.operations_log
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Recursively convert NaN, Inf, and -Inf to None for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        elif isinstance(obj, pd.Series):
            return self._make_json_safe(obj.tolist())
        else:
            return obj
    
    # ==========================================
    # PREPROCESSING
    # ==========================================
    
    def _detect_column_types(self, columns: List[str]) -> Dict[str, str]:
        """Detect numeric vs categorical columns."""
        types = {}
        for col in columns:
            if col not in self.df.columns:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                types[col] = "numeric"
            else:
                types[col] = "categorical"
        return types
    
    def _preprocess_features(
        self,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        scale: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Preprocess features: handle missing, encode categoricals, scale.
        
        Returns:
            X: Feature matrix
            y: Target vector (if target_column provided)
            info: Preprocessing information
        """
        warnings_list = []
        
        # Validate columns
        for col in feature_columns:
            if col not in self.df.columns:
                raise ValueError(f"Feature column '{col}' not found")
        
        if target_column and target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Start with clean data
        df_work = self.df[feature_columns + ([target_column] if target_column else [])].copy()
        
        # Handle missing values
        original_rows = len(df_work)
        df_work = df_work.dropna()
        dropped_rows = original_rows - len(df_work)
        
        if dropped_rows > 0:
            warnings_list.append(f"Dropped {dropped_rows} rows with missing values")
        
        if len(df_work) < 10:
            raise ValueError(f"Insufficient data after removing missing values ({len(df_work)} rows)")
        
        # Detect types and encode
        col_types = self._detect_column_types(feature_columns)
        encoding_info = {}
        
        X_parts = []
        feature_names = []
        
        for col in feature_columns:
            if col_types.get(col) == "categorical":
                # Label encoding
                unique_vals = df_work[col].unique()
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                encoded = df_work[col].map(encoding_map).values.reshape(-1, 1)
                X_parts.append(encoded)
                feature_names.append(col)
                encoding_info[col] = {"type": "label", "mapping": {str(k): v for k, v in encoding_map.items()}}
            else:
                X_parts.append(df_work[col].values.reshape(-1, 1))
                feature_names.append(col)
                encoding_info[col] = {"type": "numeric"}
        
        X = np.hstack(X_parts)
        
        # Scale features (min-max)
        scaling_info = {}
        if scale:
            X_scaled = np.zeros_like(X, dtype=float)
            for i in range(X.shape[1]):
                col_min = X[:, i].min()
                col_max = X[:, i].max()
                if col_max > col_min:
                    X_scaled[:, i] = (X[:, i] - col_min) / (col_max - col_min)
                else:
                    X_scaled[:, i] = 0
                scaling_info[feature_names[i]] = {"min": float(col_min), "max": float(col_max)}
            X = X_scaled
        
        # Target
        y = None
        target_info = {}
        if target_column:
            if col_types.get(target_column, self._detect_column_types([target_column]).get(target_column)) == "categorical":
                unique_vals = df_work[target_column].unique()
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                y = df_work[target_column].map(encoding_map).values
                target_info = {"type": "categorical", "classes": [str(v) for v in unique_vals], "mapping": {str(k): v for k, v in encoding_map.items()}}
            else:
                y = df_work[target_column].values
                target_info = {"type": "numeric"}
        
        info = {
            "n_samples": len(df_work),
            "n_features": X.shape[1],
            "feature_names": feature_names,
            "column_types": col_types,
            "encoding_info": encoding_info,
            "scaling_info": scaling_info,
            "target_info": target_info,
            "warnings": warnings_list
        }
        
        return X, y, info
    
    def _train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Manual train-test split."""
        np.random.seed(random_state)
        n = len(X)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - test_size))
        
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    def _check_class_imbalance(self, y: np.ndarray) -> Dict[str, Any]:
        """Check for class imbalance in classification."""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            "class_distribution": class_dist,
            "imbalance_ratio": float(imbalance_ratio),
            "is_imbalanced": imbalance_ratio > 3
        }
    
    def _check_multicollinearity(self, X: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Check for multicollinearity using correlation."""
        issues = []
        n_features = X.shape[1]
        
        if n_features < 2:
            return issues
        
        corr_matrix = np.corrcoef(X.T)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                if corr > 0.8:
                    issues.append({
                        "feature1": feature_names[i],
                        "feature2": feature_names[j],
                        "correlation": float(corr)
                    })
        
        return issues
    
    def _check_low_variance(self, X: np.ndarray, feature_names: List[str], threshold: float = 0.01) -> List[str]:
        """Check for low variance features."""
        low_var_features = []
        for i, name in enumerate(feature_names):
            var = np.var(X[:, i])
            if var < threshold:
                low_var_features.append(name)
        return low_var_features
    
    # ==========================================
    # REGRESSION
    # ==========================================
    
    def linear_regression(
        self,
        feature_columns: List[str],
        target_column: str,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Perform linear regression.
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            test_size: Proportion of data for testing
            
        Returns:
            Dict with model results and metrics
        """
        warnings_list = []
        
        try:
            X, y, prep_info = self._preprocess_features(feature_columns, target_column, scale=True)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        if prep_info["target_info"].get("type") == "categorical":
            return self._make_json_safe({"error": "Target must be numeric for linear regression"})
        
        # Check for issues
        multicollinearity = self._check_multicollinearity(X, prep_info["feature_names"])
        if multicollinearity:
            warnings_list.append(f"Multicollinearity detected: {len(multicollinearity)} feature pairs with |r| > 0.8")
        
        low_var = self._check_low_variance(X, prep_info["feature_names"])
        if low_var:
            warnings_list.append(f"Low variance features: {low_var}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        
        # Fit linear regression using OLS
        # Add bias term
        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]
        
        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X_train_b.T @ X_train_b)
            coefficients = XtX_inv @ X_train_b.T @ y_train
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            coefficients = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
            warnings_list.append("Used pseudo-inverse due to singular matrix")
        
        intercept = coefficients[0]
        coefs = coefficients[1:]
        
        # Predictions
        y_train_pred = X_train_b @ coefficients
        y_test_pred = X_test_b @ coefficients
        
        # Metrics
        def calc_metrics(y_true, y_pred):
            residuals = y_true - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        
        train_metrics = calc_metrics(y_train, y_train_pred)
        test_metrics = calc_metrics(y_test, y_test_pred)
        
        # Check for overfitting
        if train_metrics["r2"] - test_metrics["r2"] > 0.2:
            warnings_list.append("Possible overfitting: train R² much higher than test R²")
        
        # Feature importance (standardized coefficients)
        feature_importance = {}
        X_std = np.std(X_train, axis=0)
        y_std = np.std(y_train)
        for i, name in enumerate(prep_info["feature_names"]):
            if y_std > 0 and X_std[i] > 0:
                std_coef = coefs[i] * X_std[i] / y_std
            else:
                std_coef = 0
            feature_importance[name] = {
                "coefficient": float(coefs[i]),
                "standardized_coefficient": float(std_coef)
            }
        
        result = {
            "model": "linear_regression",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "intercept": float(intercept),
            "coefficients": {name: float(coefs[i]) for i, name in enumerate(prep_info["feature_names"])},
            "feature_importance": feature_importance,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_size": test_size,
            "preprocessing": prep_info,
            "diagnostics": {
                "multicollinearity": multicollinearity,
                "low_variance_features": low_var
            },
            "warnings": warnings_list
        }
        
        self._log_operation("linear_regression", {
            "n_features": X.shape[1],
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    # ==========================================
    # CLASSIFICATION
    # ==========================================
    
    def logistic_regression(
        self,
        feature_columns: List[str],
        target_column: str,
        test_size: float = 0.2,
        max_iter: int = 1000,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform logistic regression (binary classification).
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            test_size: Proportion of data for testing
            max_iter: Maximum iterations for gradient descent
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Dict with model results and metrics
        """
        warnings_list = []
        
        try:
            X, y, prep_info = self._preprocess_features(feature_columns, target_column, scale=True)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        # Check if binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            return self._make_json_safe({
                "error": f"Logistic regression requires binary target, found {len(unique_classes)} classes"
            })
        
        # Check class imbalance
        imbalance_info = self._check_class_imbalance(y)
        if imbalance_info["is_imbalanced"]:
            warnings_list.append(f"Class imbalance detected (ratio: {imbalance_info['imbalance_ratio']:.2f})")
        
        # Train-test split
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        
        # Add bias
        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]
        
        # Sigmoid function
        def sigmoid(z):
            z = np.clip(z, -500, 500)  # Prevent overflow
            return 1 / (1 + np.exp(-z))
        
        # Gradient descent
        n_features = X_train_b.shape[1]
        weights = np.zeros(n_features)
        
        for _ in range(max_iter):
            z = X_train_b @ weights
            predictions = sigmoid(z)
            gradient = X_train_b.T @ (predictions - y_train) / len(y_train)
            weights -= learning_rate * gradient
        
        intercept = weights[0]
        coefs = weights[1:]
        
        # Predictions
        y_train_prob = sigmoid(X_train_b @ weights)
        y_test_prob = sigmoid(X_test_b @ weights)
        
        y_train_pred = (y_train_prob >= 0.5).astype(int)
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        
        # Metrics
        def calc_classification_metrics(y_true, y_pred):
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": {
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
                }
            }
        
        train_metrics = calc_classification_metrics(y_train, y_train_pred)
        test_metrics = calc_classification_metrics(y_test, y_test_pred)
        
        # Check for overfitting
        if train_metrics["accuracy"] - test_metrics["accuracy"] > 0.15:
            warnings_list.append("Possible overfitting: train accuracy much higher than test accuracy")
        
        result = {
            "model": "logistic_regression",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "classes": prep_info["target_info"].get("classes", [str(c) for c in unique_classes]),
            "intercept": float(intercept),
            "coefficients": {name: float(coefs[i]) for i, name in enumerate(prep_info["feature_names"])},
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_size": test_size,
            "class_balance": imbalance_info,
            "preprocessing": prep_info,
            "warnings": warnings_list
        }
        
        self._log_operation("logistic_regression", {
            "n_features": X.shape[1],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"]
        })
        
        return self._make_json_safe(result)
    
    def random_forest_classifier(
        self,
        feature_columns: List[str],
        target_column: str,
        n_trees: int = 10,
        max_depth: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Perform Random Forest classification (simplified implementation).
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            test_size: Proportion of data for testing
            
        Returns:
            Dict with model results and metrics
        """
        warnings_list = []
        
        try:
            X, y, prep_info = self._preprocess_features(feature_columns, target_column, scale=False)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Check class imbalance
        imbalance_info = self._check_class_imbalance(y)
        if imbalance_info["is_imbalanced"]:
            warnings_list.append(f"Class imbalance detected (ratio: {imbalance_info['imbalance_ratio']:.2f})")
        
        # Train-test split
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        
        # Simple decision tree implementation
        def build_tree(X, y, depth=0):
            n_samples, n_features = X.shape
            unique_classes_node = np.unique(y)
            
            # Stopping criteria
            if depth >= max_depth or len(unique_classes_node) == 1 or n_samples < 5:
                counts = np.bincount(y.astype(int), minlength=n_classes)
                return {"type": "leaf", "class": int(np.argmax(counts)), "probs": (counts / counts.sum()).tolist()}
            
            # Find best split
            best_gini = float('inf')
            best_split = None
            
            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds[:-1]:
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                        continue
                    
                    # Gini impurity
                    def gini(y_subset):
                        if len(y_subset) == 0:
                            return 0
                        counts = np.bincount(y_subset.astype(int), minlength=n_classes)
                        probs = counts / len(y_subset)
                        return 1 - np.sum(probs ** 2)
                    
                    gini_left = gini(y[left_mask])
                    gini_right = gini(y[right_mask])
                    weighted_gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / n_samples
                    
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_split = {"feature": feature, "threshold": threshold}
            
            if best_split is None:
                counts = np.bincount(y.astype(int), minlength=n_classes)
                return {"type": "leaf", "class": int(np.argmax(counts)), "probs": (counts / counts.sum()).tolist()}
            
            left_mask = X[:, best_split["feature"]] <= best_split["threshold"]
            
            return {
                "type": "node",
                "feature": best_split["feature"],
                "threshold": float(best_split["threshold"]),
                "left": build_tree(X[left_mask], y[left_mask], depth + 1),
                "right": build_tree(X[~left_mask], y[~left_mask], depth + 1)
            }
        
        def predict_tree(tree, x):
            if tree["type"] == "leaf":
                return tree["probs"]
            if x[tree["feature"]] <= tree["threshold"]:
                return predict_tree(tree["left"], x)
            return predict_tree(tree["right"], x)
        
        # Build forest with bootstrap sampling
        trees = []
        feature_importances = np.zeros(X.shape[1])
        
        for i in range(n_trees):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            tree = build_tree(X_boot, y_boot)
            trees.append(tree)
        
        # Predictions
        def predict_forest(X):
            all_probs = []
            for x in X:
                probs = np.zeros(n_classes)
                for tree in trees:
                    probs += np.array(predict_tree(tree, x))
                probs /= n_trees
                all_probs.append(probs)
            return np.array(all_probs)
        
        train_probs = predict_forest(X_train)
        test_probs = predict_forest(X_test)
        
        y_train_pred = np.argmax(train_probs, axis=1)
        y_test_pred = np.argmax(test_probs, axis=1)
        
        # Metrics
        def calc_multiclass_metrics(y_true, y_pred, n_classes):
            accuracy = np.mean(y_true == y_pred)
            
            # Per-class metrics
            precision_sum = 0
            recall_sum = 0
            
            for c in range(n_classes):
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                
                precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precision_sum += precision_c
                recall_sum += recall_c
            
            precision = precision_sum / n_classes
            recall = recall_sum / n_classes
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        
        train_metrics = calc_multiclass_metrics(y_train, y_train_pred, n_classes)
        test_metrics = calc_multiclass_metrics(y_test, y_test_pred, n_classes)
        
        # Check for overfitting
        if train_metrics["accuracy"] - test_metrics["accuracy"] > 0.15:
            warnings_list.append("Possible overfitting detected")
        
        result = {
            "model": "random_forest_classifier",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_trees": n_trees,
            "max_depth": max_depth,
            "n_classes": n_classes,
            "classes": prep_info["target_info"].get("classes", [str(c) for c in unique_classes]),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_size": test_size,
            "class_balance": imbalance_info,
            "preprocessing": prep_info,
            "warnings": warnings_list
        }
        
        self._log_operation("random_forest_classifier", {
            "n_trees": n_trees,
            "max_depth": max_depth,
            "test_accuracy": test_metrics["accuracy"]
        })
        
        return self._make_json_safe(result)
    
    def random_forest_regressor(
        self,
        feature_columns: List[str],
        target_column: str,
        n_trees: int = 10,
        max_depth: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Perform Random Forest regression (simplified implementation).
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of each tree
            test_size: Proportion of data for testing
            
        Returns:
            Dict with model results and metrics
        """
        warnings_list = []
        
        try:
            X, y, prep_info = self._preprocess_features(feature_columns, target_column, scale=False)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        if prep_info["target_info"].get("type") == "categorical":
            return self._make_json_safe({"error": "Target must be numeric for regression"})
        
        # Train-test split
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, test_size)
        
        # Simple regression tree implementation
        def build_tree(X, y, depth=0):
            n_samples = len(X)
            
            # Stopping criteria
            if depth >= max_depth or n_samples < 5 or np.std(y) < 1e-6:
                return {"type": "leaf", "value": float(np.mean(y))}
            
            # Find best split
            best_mse = float('inf')
            best_split = None
            
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds[:-1]:
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                        continue
                    
                    mse_left = np.var(y[left_mask]) * np.sum(left_mask)
                    mse_right = np.var(y[right_mask]) * np.sum(right_mask)
                    total_mse = (mse_left + mse_right) / n_samples
                    
                    if total_mse < best_mse:
                        best_mse = total_mse
                        best_split = {"feature": feature, "threshold": threshold}
            
            if best_split is None:
                return {"type": "leaf", "value": float(np.mean(y))}
            
            left_mask = X[:, best_split["feature"]] <= best_split["threshold"]
            
            return {
                "type": "node",
                "feature": best_split["feature"],
                "threshold": float(best_split["threshold"]),
                "left": build_tree(X[left_mask], y[left_mask], depth + 1),
                "right": build_tree(X[~left_mask], y[~left_mask], depth + 1)
            }
        
        def predict_tree(tree, x):
            if tree["type"] == "leaf":
                return tree["value"]
            if x[tree["feature"]] <= tree["threshold"]:
                return predict_tree(tree["left"], x)
            return predict_tree(tree["right"], x)
        
        # Build forest
        trees = []
        for _ in range(n_trees):
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            tree = build_tree(X_train[indices], y_train[indices])
            trees.append(tree)
        
        # Predictions
        def predict_forest(X):
            predictions = np.zeros(len(X))
            for x_idx, x in enumerate(X):
                preds = [predict_tree(tree, x) for tree in trees]
                predictions[x_idx] = np.mean(preds)
            return predictions
        
        y_train_pred = predict_forest(X_train)
        y_test_pred = predict_forest(X_test)
        
        # Metrics
        def calc_metrics(y_true, y_pred):
            residuals = y_true - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            return {
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "mae": float(np.mean(np.abs(residuals))),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
            }
        
        train_metrics = calc_metrics(y_train, y_train_pred)
        test_metrics = calc_metrics(y_test, y_test_pred)
        
        if train_metrics["r2"] - test_metrics["r2"] > 0.2:
            warnings_list.append("Possible overfitting detected")
        
        result = {
            "model": "random_forest_regressor",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_trees": n_trees,
            "max_depth": max_depth,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_size": test_size,
            "preprocessing": prep_info,
            "warnings": warnings_list
        }
        
        self._log_operation("random_forest_regressor", {
            "n_trees": n_trees,
            "test_r2": test_metrics["r2"],
            "test_rmse": test_metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    # ==========================================
    # CLUSTERING
    # ==========================================
    
    def kmeans(
        self,
        feature_columns: List[str],
        n_clusters: int = 3,
        max_iter: int = 100
    ) -> Dict[str, Any]:
        """
        Perform K-Means clustering.
        
        Args:
            feature_columns: List of feature column names
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            
        Returns:
            Dict with clustering results
        """
        warnings_list = []
        
        try:
            X, _, prep_info = self._preprocess_features(feature_columns, scale=True)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        n_samples = len(X)
        
        if n_samples < n_clusters:
            return self._make_json_safe({"error": f"Need at least {n_clusters} samples for {n_clusters} clusters"})
        
        # Initialize centroids randomly
        np.random.seed(42)
        initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[initial_indices].copy()
        
        # K-Means algorithm
        labels = np.zeros(n_samples, dtype=int)
        
        for iteration in range(max_iter):
            # Assign clusters
            old_labels = labels.copy()
            for i in range(n_samples):
                distances = np.sqrt(np.sum((centroids - X[i]) ** 2, axis=1))
                labels[i] = np.argmin(distances)
            
            # Check convergence
            if np.all(old_labels == labels):
                break
            
            # Update centroids
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centroids[k] = np.mean(cluster_points, axis=0)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        
        # Cluster statistics
        cluster_stats = {}
        for k in range(n_clusters):
            cluster_mask = labels == k
            cluster_size = np.sum(cluster_mask)
            
            cluster_stats[f"cluster_{k}"] = {
                "size": int(cluster_size),
                "percentage": float(cluster_size / n_samples * 100),
                "centroid": centroids[k].tolist()
            }
        
        # Silhouette score (simplified)
        silhouette_scores = []
        for i in range(min(1000, n_samples)):  # Sample for speed
            own_cluster = labels[i]
            own_cluster_points = X[labels == own_cluster]
            
            if len(own_cluster_points) > 1:
                a = np.mean(np.sqrt(np.sum((own_cluster_points - X[i]) ** 2, axis=1)))
            else:
                a = 0
            
            b = float('inf')
            for k in range(n_clusters):
                if k != own_cluster:
                    other_cluster_points = X[labels == k]
                    if len(other_cluster_points) > 0:
                        avg_dist = np.mean(np.sqrt(np.sum((other_cluster_points - X[i]) ** 2, axis=1)))
                        b = min(b, avg_dist)
            
            if b == float('inf'):
                b = 0
            
            if max(a, b) > 0:
                silhouette_scores.append((b - a) / max(a, b))
        
        avg_silhouette = float(np.mean(silhouette_scores)) if silhouette_scores else 0
        
        result = {
            "model": "kmeans",
            "n_samples": n_samples,
            "n_features": X.shape[1],
            "n_clusters": n_clusters,
            "iterations": iteration + 1,
            "inertia": float(inertia),
            "silhouette_score": avg_silhouette,
            "cluster_centers": centroids.tolist(),
            "labels": labels.tolist(),
            "cluster_statistics": cluster_stats,
            "preprocessing": prep_info,
            "warnings": warnings_list
        }
        
        self._log_operation("kmeans", {
            "n_clusters": n_clusters,
            "inertia": float(inertia),
            "silhouette": avg_silhouette
        })
        
        return self._make_json_safe(result)
    
    # ==========================================
    # PCA
    # ==========================================
    
    def pca(
        self,
        feature_columns: List[str],
        n_components: int = 2
    ) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis.
        
        Args:
            feature_columns: List of feature column names
            n_components: Number of components to extract
            
        Returns:
            Dict with PCA results
        """
        warnings_list = []
        
        try:
            X, _, prep_info = self._preprocess_features(feature_columns, scale=True)
            warnings_list.extend(prep_info.get("warnings", []))
        except ValueError as e:
            return self._make_json_safe({"error": str(e)})
        
        n_samples, n_features = X.shape
        n_components = min(n_components, n_features, n_samples)
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select top components
        components = eigenvectors[:, :n_components].T
        eigenvalues_selected = eigenvalues[:n_components]
        
        # Transform data
        X_transformed = X_centered @ components.T
        
        # Explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues_selected / total_variance if total_variance > 0 else eigenvalues_selected
        cumulative_variance = np.cumsum(explained_variance)
        
        # Feature loadings
        loadings = {}
        for i, name in enumerate(prep_info["feature_names"]):
            loadings[name] = {
                f"PC{j+1}": float(components[j, i])
                for j in range(n_components)
            }
        
        result = {
            "model": "pca",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components,
            "explained_variance_ratio": explained_variance.tolist(),
            "cumulative_variance_ratio": cumulative_variance.tolist(),
            "components": components.tolist(),
            "loadings": loadings,
            "transformed_data": X_transformed.tolist(),
            "preprocessing": prep_info,
            "warnings": warnings_list
        }
        
        self._log_operation("pca", {
            "n_components": n_components,
            "explained_variance": float(cumulative_variance[-1]) if len(cumulative_variance) > 0 else 0
        })
        
        return self._make_json_safe(result)
    
    @staticmethod
    def ml_multiple(
        file_ids: List[str],
        file_manager: Any,
        ml_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run ML operations on multiple files independently
        
        Args:
            file_ids: List of file identifiers
            file_manager: FileManager instance to load files
            ml_type: Type of ML operation ("classify", "regress", "cluster", "pca", "feature_importance")
            params: Parameters for the ML operation
            
        Returns:
            Dictionary mapping file_id to ML results:
            {
                "<file_id>": {"result": {...}, "status": "ok"},
                "<file_id>": {"error": "..."}
            }
        """
        results = {}
        
        for file_id in file_ids:
            try:
                # Load file
                file_path = file_manager.get_file_path(file_id)
                if not file_path:
                    results[file_id] = {"error": "File not found"}
                    continue
                
                df = file_manager.load_dataframe(file_path)
                
                # Initialize engine for this file (isolated preprocessing)
                engine = MLEngine(df)
                
                # Perform ML operation
                if ml_type == "classify":
                    result = engine.logistic_regression(
                        target_column=params.get("target_column"),
                        feature_columns=params.get("feature_columns", []),
                        test_size=params.get("test_size", 0.2)
                    )
                elif ml_type == "regress":
                    result = engine.linear_regression(
                        target_column=params.get("target_column"),
                        feature_columns=params.get("feature_columns", []),
                        test_size=params.get("test_size", 0.2)
                    )
                elif ml_type == "cluster":
                    result = engine.kmeans_clustering(
                        feature_columns=params.get("feature_columns", []),
                        n_clusters=params.get("n_clusters", 3)
                    )
                elif ml_type == "pca":
                    result = engine.pca(
                        feature_columns=params.get("feature_columns", []),
                        n_components=params.get("n_components")
                    )
                elif ml_type == "feature_importance":
                    result = engine.random_forest_regression(
                        target_column=params.get("target_column"),
                        feature_columns=params.get("feature_columns", []),
                        n_estimators=params.get("n_estimators", 100),
                        test_size=params.get("test_size", 0.2)
                    )
                else:
                    results[file_id] = {"error": f"Unknown ML type: {ml_type}"}
                    continue
                
                results[file_id] = {
                    "result": result,
                    "operations_log": engine.operations_log,
                    "status": "ok"
                }
                
            except Exception as e:
                results[file_id] = {"error": str(e)}
        
        return results
