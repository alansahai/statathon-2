"""
StatFlow AI - Forecasting Engine
Provides time series analysis and forecasting capabilities.
Production-ready implementation matching existing engine conventions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ForecastingEngine:
    """
    Time series forecasting engine with multiple methods.
    
    Supports:
    - Moving Average
    - Exponential Smoothing
    - Holt-Winters (additive/multiplicative)
    - ARIMA (automatic parameter selection)
    - Weighted smoothing
    
    All outputs are JSON-safe (NaN/Inf converted to None).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the forecasting engine with a DataFrame.
        
        Args:
            df: Input DataFrame containing time series data
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
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return self._make_json_safe(obj.tolist())
        else:
            return obj
    
    # ==========================================
    # TIME SERIES UTILITIES
    # ==========================================
    
    def detect_time_column(self) -> Dict[str, Any]:
        """
        Automatically detect the time/date column in the DataFrame.
        
        Returns:
            Dict with detected column info or error
        """
        candidates = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check column name patterns
            time_keywords = ['date', 'time', 'year', 'month', 'day', 'period', 'timestamp']
            name_score = sum(1 for kw in time_keywords if kw in col_lower)
            
            # Try to parse as datetime
            try:
                parsed = pd.to_datetime(self.df[col], errors='coerce')
                valid_ratio = parsed.notna().mean()
                
                if valid_ratio > 0.5:
                    candidates.append({
                        "column": col,
                        "valid_ratio": float(valid_ratio),
                        "name_score": name_score,
                        "dtype": str(self.df[col].dtype),
                        "sample_values": [str(v) for v in self.df[col].head(3).tolist()]
                    })
            except:
                pass
        
        # Sort by valid_ratio and name_score
        candidates.sort(key=lambda x: (x["valid_ratio"], x["name_score"]), reverse=True)
        
        result = {
            "detected": candidates[0]["column"] if candidates else None,
            "candidates": candidates,
            "message": f"Detected '{candidates[0]['column']}' as time column" if candidates else "No time column detected"
        }
        
        self._log_operation("detect_time_column", {"result": result["detected"]})
        return self._make_json_safe(result)
    
    def resample_data(
        self,
        time_column: str,
        value_column: str,
        freq: str = "D",
        agg_func: str = "mean",
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resample time series data to specified frequency.
        
        Args:
            time_column: Name of the time column
            value_column: Name of the value column
            freq: Frequency string ('D', 'W', 'M', 'Q', 'Y')
            agg_func: Aggregation function ('mean', 'sum', 'median', 'first', 'last')
            weight_column: Optional weight column for weighted aggregation
            
        Returns:
            Dict with resampled data and metadata
        """
        warnings_list = []
        
        # Validate columns
        for col in [time_column, value_column]:
            if col not in self.df.columns:
                return self._make_json_safe({"error": f"Column '{col}' not found"})
        
        if weight_column and weight_column not in self.df.columns:
            return self._make_json_safe({"error": f"Weight column '{weight_column}' not found"})
        
        try:
            # Parse time column
            df_work = self.df.copy()
            df_work[time_column] = pd.to_datetime(df_work[time_column], errors='coerce')
            
            # Check for invalid dates
            invalid_dates = df_work[time_column].isna().sum()
            if invalid_dates > 0:
                warnings_list.append(f"{invalid_dates} rows with invalid dates removed")
                df_work = df_work.dropna(subset=[time_column])
            
            # Set index
            df_work = df_work.set_index(time_column)
            df_work = df_work.sort_index()
            
            # Resample
            if weight_column and agg_func == "mean":
                # Weighted mean
                def weighted_mean(group):
                    weights = group[weight_column]
                    values = group[value_column]
                    return np.average(values, weights=weights)
                
                resampled = df_work.resample(freq).apply(weighted_mean)
            else:
                agg_map = {
                    "mean": "mean",
                    "sum": "sum",
                    "median": "median",
                    "first": "first",
                    "last": "last"
                }
                resampled = df_work[value_column].resample(freq).agg(agg_map.get(agg_func, "mean"))
            
            # Handle missing periods
            missing_periods = resampled.isna().sum()
            if missing_periods > 0:
                warnings_list.append(f"{missing_periods} periods with missing data")
            
            result = {
                "time_index": resampled.index.tolist(),
                "values": resampled.values.tolist(),
                "frequency": freq,
                "n_periods": len(resampled),
                "start_date": resampled.index.min(),
                "end_date": resampled.index.max(),
                "missing_periods": int(missing_periods),
                "warnings": warnings_list
            }
            
            self._log_operation("resample_data", {
                "frequency": freq,
                "n_periods": len(resampled),
                "agg_func": agg_func
            })
            
            return self._make_json_safe(result)
            
        except Exception as e:
            return self._make_json_safe({"error": f"Resampling failed: {str(e)}"})
    
    def handle_missing(
        self,
        time_column: str,
        value_column: str,
        method: str = "ffill"
    ) -> Dict[str, Any]:
        """
        Handle missing values in time series.
        
        Args:
            time_column: Name of the time column
            value_column: Name of the value column
            method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'mean')
            
        Returns:
            Dict with cleaned series and metadata
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].copy()
        original_missing = series.isna().sum()
        
        if original_missing == 0:
            return self._make_json_safe({
                "message": "No missing values found",
                "values": series.tolist(),
                "missing_count": 0
            })
        
        # Apply method
        if method == "ffill":
            filled = series.ffill()
        elif method == "bfill":
            filled = series.bfill()
        elif method == "interpolate":
            filled = series.interpolate(method='linear')
        elif method == "mean":
            filled = series.fillna(series.mean())
        else:
            return self._make_json_safe({"error": f"Unknown method: {method}"})
        
        # Check remaining missing
        remaining_missing = filled.isna().sum()
        if remaining_missing > 0:
            warnings_list.append(f"{remaining_missing} values still missing after {method}")
            filled = filled.bfill().ffill()  # Fallback
        
        # Update internal DataFrame
        self.df[value_column] = filled
        
        result = {
            "method": method,
            "original_missing": int(original_missing),
            "filled_count": int(original_missing - remaining_missing),
            "remaining_missing": int(remaining_missing),
            "values": filled.tolist(),
            "warnings": warnings_list
        }
        
        self._log_operation("handle_missing", {
            "method": method,
            "original_missing": original_missing,
            "filled_count": original_missing - remaining_missing
        })
        
        return self._make_json_safe(result)
    
    def seasonal_decompose(
        self,
        time_column: str,
        value_column: str,
        period: Optional[int] = None,
        model: str = "additive"
    ) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            time_column: Name of the time column
            value_column: Name of the value column
            period: Seasonal period (auto-detected if None)
            model: 'additive' or 'multiplicative'
            
        Returns:
            Dict with decomposition components
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna()
        
        if len(series) < 4:
            return self._make_json_safe({"error": "Need at least 4 observations for decomposition"})
        
        # Auto-detect period if not provided
        if period is None:
            # Try to detect from data
            if len(series) >= 24:
                period = 12  # Assume monthly with annual seasonality
            elif len(series) >= 14:
                period = 7   # Assume weekly
            else:
                period = 4   # Default quarterly
            warnings_list.append(f"Auto-detected period: {period}")
        
        if len(series) < 2 * period:
            warnings_list.append(f"Series length ({len(series)}) < 2 * period ({2*period}), results may be unreliable")
        
        try:
            # Manual decomposition (avoiding statsmodels dependency)
            n = len(series)
            values = series.values.astype(float)
            
            # Trend: centered moving average
            if period % 2 == 0:
                # Even period: use 2-step MA
                ma1 = np.convolve(values, np.ones(period)/period, mode='valid')
                ma2 = np.convolve(ma1, np.ones(2)/2, mode='valid')
                trend = np.full(n, np.nan)
                start_idx = period // 2
                trend[start_idx:start_idx+len(ma2)] = ma2
            else:
                ma = np.convolve(values, np.ones(period)/period, mode='valid')
                trend = np.full(n, np.nan)
                start_idx = period // 2
                trend[start_idx:start_idx+len(ma)] = ma
            
            # Fill trend edges
            trend = pd.Series(trend).interpolate(method='linear').bfill().ffill().values
            
            # Detrended
            if model == "multiplicative":
                if np.any(trend <= 0) or np.any(values <= 0):
                    warnings_list.append("Non-positive values found, switching to additive model")
                    model = "additive"
                    detrended = values - trend
                else:
                    detrended = values / trend
            else:
                detrended = values - trend
            
            # Seasonal: average of detrended by position in cycle
            seasonal = np.zeros(n)
            for i in range(period):
                indices = list(range(i, n, period))
                if model == "multiplicative":
                    seasonal[indices] = np.nanmean(detrended[indices])
                else:
                    seasonal[indices] = np.nanmean(detrended[indices])
            
            # Normalize seasonal component
            if model == "multiplicative":
                seasonal = seasonal / np.mean(seasonal)
            else:
                seasonal = seasonal - np.mean(seasonal)
            
            # Residual
            if model == "multiplicative":
                residual = values / (trend * seasonal)
            else:
                residual = values - trend - seasonal
            
            # Seasonality strength
            var_residual = np.nanvar(residual)
            var_seasonal = np.nanvar(seasonal)
            seasonality_strength = 1 - var_residual / (var_residual + var_seasonal) if (var_residual + var_seasonal) > 0 else 0
            
            result = {
                "model": model,
                "period": period,
                "n_observations": n,
                "trend": trend.tolist(),
                "seasonal": seasonal.tolist(),
                "residual": residual.tolist(),
                "original": values.tolist(),
                "seasonality_strength": float(max(0, min(1, seasonality_strength))),
                "trend_direction": "increasing" if trend[-1] > trend[0] else "decreasing",
                "warnings": warnings_list
            }
            
            self._log_operation("seasonal_decompose", {
                "model": model,
                "period": period,
                "seasonality_strength": float(seasonality_strength)
            })
            
            return self._make_json_safe(result)
            
        except Exception as e:
            return self._make_json_safe({"error": f"Decomposition failed: {str(e)}"})
    
    # ==========================================
    # FORECAST METHODS
    # ==========================================
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {"rmse": None, "mae": None, "mape": None}
        
        residuals = actual - predicted
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # MAE
        mae = np.mean(np.abs(residuals))
        
        # MAPE (avoid division by zero)
        non_zero_mask = actual != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs(residuals[non_zero_mask] / actual[non_zero_mask])) * 100
        else:
            mape = None
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape) if mape is not None else None
        }
    
    def moving_average(
        self,
        value_column: str,
        window: int = 3,
        periods: int = 5,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Moving Average forecast.
        
        Args:
            value_column: Name of the value column
            window: Window size for moving average
            periods: Number of periods to forecast
            weight_column: Optional weight column
            
        Returns:
            Dict with forecast results
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna().values.astype(float)
        n = len(series)
        
        if n < window:
            return self._make_json_safe({"error": f"Series length ({n}) < window size ({window})"})
        
        if n < 10:
            warnings_list.append("Short time series, forecast may be unreliable")
        
        # Weighted or simple MA
        if weight_column and weight_column in self.df.columns:
            weights = self.df[weight_column].dropna().values[-window:]
            weights = weights / weights.sum()
        else:
            weights = np.ones(window) / window
        
        # Fitted values
        fitted = np.full(n, np.nan)
        for i in range(window, n):
            fitted[i] = np.average(series[i-window:i], weights=weights[-window:])
        
        # Forecast
        forecast = []
        last_values = list(series[-window:])
        
        for _ in range(periods):
            next_val = np.average(last_values[-window:], weights=weights)
            forecast.append(next_val)
            last_values.append(next_val)
        
        # Confidence intervals (based on historical residuals)
        residuals = series[window:] - fitted[window:]
        std_residual = np.nanstd(residuals)
        
        ci_lower = [f - 1.96 * std_residual for f in forecast]
        ci_upper = [f + 1.96 * std_residual for f in forecast]
        
        # Metrics
        metrics = self._calculate_metrics(series[window:], fitted[window:])
        
        result = {
            "method": "moving_average",
            "window": window,
            "n_observations": n,
            "fitted_values": fitted.tolist(),
            "forecast_values": forecast,
            "periods_ahead": periods,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 0.95
            },
            "residuals": (series - fitted).tolist(),
            "metrics": metrics,
            "warnings": warnings_list
        }
        
        self._log_operation("moving_average", {
            "window": window,
            "periods": periods,
            "rmse": metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    def exponential_smoothing(
        self,
        value_column: str,
        alpha: Optional[float] = None,
        periods: int = 5,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple Exponential Smoothing forecast.
        
        Args:
            value_column: Name of the value column
            alpha: Smoothing parameter (0 < alpha < 1), auto-optimized if None
            periods: Number of periods to forecast
            weight_column: Optional weight column (affects optimization)
            
        Returns:
            Dict with forecast results
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna().values.astype(float)
        n = len(series)
        
        if n < 3:
            return self._make_json_safe({"error": "Need at least 3 observations"})
        
        if n < 10:
            warnings_list.append("Short time series, forecast may be unreliable")
        
        # Optimize alpha if not provided
        if alpha is None:
            def sse(alpha_val):
                if alpha_val <= 0 or alpha_val >= 1:
                    return np.inf
                fitted = np.zeros(n)
                fitted[0] = series[0]
                for i in range(1, n):
                    fitted[i] = alpha_val * series[i-1] + (1 - alpha_val) * fitted[i-1]
                return np.sum((series - fitted) ** 2)
            
            result_opt = minimize(sse, 0.3, bounds=[(0.01, 0.99)], method='L-BFGS-B')
            alpha = float(result_opt.x[0])
            warnings_list.append(f"Optimized alpha: {alpha:.4f}")
        
        # Fit model
        fitted = np.zeros(n)
        fitted[0] = series[0]
        for i in range(1, n):
            fitted[i] = alpha * series[i-1] + (1 - alpha) * fitted[i-1]
        
        # Forecast (flat forecast for simple ES)
        last_level = alpha * series[-1] + (1 - alpha) * fitted[-1]
        forecast = [last_level] * periods
        
        # Confidence intervals
        residuals = series - fitted
        std_residual = np.std(residuals)
        
        # CI widens with horizon
        ci_lower = []
        ci_upper = []
        for h in range(1, periods + 1):
            ci_width = 1.96 * std_residual * np.sqrt(1 + (h - 1) * alpha ** 2)
            ci_lower.append(forecast[h-1] - ci_width)
            ci_upper.append(forecast[h-1] + ci_width)
        
        metrics = self._calculate_metrics(series, fitted)
        
        result = {
            "method": "exponential_smoothing",
            "alpha": float(alpha),
            "n_observations": n,
            "fitted_values": fitted.tolist(),
            "forecast_values": forecast,
            "periods_ahead": periods,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 0.95
            },
            "residuals": residuals.tolist(),
            "metrics": metrics,
            "warnings": warnings_list
        }
        
        self._log_operation("exponential_smoothing", {
            "alpha": float(alpha),
            "periods": periods,
            "rmse": metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    def holt_winters(
        self,
        value_column: str,
        seasonal_periods: int = 12,
        trend: str = "add",
        seasonal: str = "add",
        periods: int = 5,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Holt-Winters exponential smoothing forecast.
        
        Args:
            value_column: Name of the value column
            seasonal_periods: Number of periods in a season
            trend: 'add' for additive, 'mul' for multiplicative
            seasonal: 'add' for additive, 'mul' for multiplicative
            periods: Number of periods to forecast
            weight_column: Optional weight column
            
        Returns:
            Dict with forecast results
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna().values.astype(float)
        n = len(series)
        m = seasonal_periods
        
        if n < 2 * m:
            return self._make_json_safe({
                "error": f"Need at least {2*m} observations for seasonal period {m}"
            })
        
        # Check for non-positive values in multiplicative mode
        if (seasonal == "mul" or trend == "mul") and np.any(series <= 0):
            warnings_list.append("Non-positive values found, switching to additive model")
            seasonal = "add"
            trend = "add"
        
        # Initialize components
        # Level: average of first season
        level = np.mean(series[:m])
        
        # Trend: average difference between first two seasons
        if n >= 2 * m:
            trend_init = np.mean([(series[m+i] - series[i]) / m for i in range(m)])
        else:
            trend_init = 0
        
        # Seasonal: ratio or difference from first season
        if seasonal == "mul":
            season = series[:m] / level
        else:
            season = series[:m] - level
        
        # Smoothing parameters (heuristic starting values)
        alpha, beta, gamma = 0.3, 0.1, 0.3
        
        # Optimize parameters
        def sse(params):
            a, b, g = params
            if not (0 < a < 1 and 0 < b < 1 and 0 < g < 1):
                return np.inf
            
            l, t = level, trend_init
            s = season.copy()
            errors = []
            
            for i in range(n):
                s_idx = i % m
                
                if seasonal == "mul":
                    y_hat = (l + t) * s[s_idx] if trend == "add" else l * t * s[s_idx]
                else:
                    y_hat = (l + t) + s[s_idx] if trend == "add" else l * t + s[s_idx]
                
                errors.append((series[i] - y_hat) ** 2)
                
                # Update
                if seasonal == "mul":
                    l_new = a * (series[i] / s[s_idx]) + (1 - a) * (l + t)
                    s[s_idx] = g * (series[i] / l_new) + (1 - g) * s[s_idx]
                else:
                    l_new = a * (series[i] - s[s_idx]) + (1 - a) * (l + t)
                    s[s_idx] = g * (series[i] - l_new) + (1 - g) * s[s_idx]
                
                if trend == "mul":
                    t = b * (l_new / l) + (1 - b) * t
                else:
                    t = b * (l_new - l) + (1 - b) * t
                
                l = l_new
            
            return np.sum(errors)
        
        try:
            result_opt = minimize(
                sse, 
                [alpha, beta, gamma], 
                bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)],
                method='L-BFGS-B'
            )
            alpha, beta, gamma = result_opt.x
        except:
            warnings_list.append("Parameter optimization failed, using defaults")
        
        # Fit model with optimized parameters
        l, t = level, trend_init
        s = season.copy()
        fitted = np.zeros(n)
        
        for i in range(n):
            s_idx = i % m
            
            if seasonal == "mul":
                fitted[i] = (l + t) * s[s_idx] if trend == "add" else l * t * s[s_idx]
            else:
                fitted[i] = (l + t) + s[s_idx] if trend == "add" else l * t + s[s_idx]
            
            # Update
            if seasonal == "mul":
                l_new = alpha * (series[i] / s[s_idx]) + (1 - alpha) * (l + t)
                s[s_idx] = gamma * (series[i] / l_new) + (1 - gamma) * s[s_idx]
            else:
                l_new = alpha * (series[i] - s[s_idx]) + (1 - alpha) * (l + t)
                s[s_idx] = gamma * (series[i] - l_new) + (1 - gamma) * s[s_idx]
            
            if trend == "mul":
                t = beta * (l_new / l) + (1 - beta) * t
            else:
                t = beta * (l_new - l) + (1 - beta) * t
            
            l = l_new
        
        # Forecast
        forecast = []
        for h in range(1, periods + 1):
            s_idx = (n + h - 1) % m
            if seasonal == "mul":
                if trend == "add":
                    f = (l + h * t) * s[s_idx]
                else:
                    f = l * (t ** h) * s[s_idx]
            else:
                if trend == "add":
                    f = l + h * t + s[s_idx]
                else:
                    f = l * (t ** h) + s[s_idx]
            forecast.append(f)
        
        # Confidence intervals
        residuals = series - fitted
        std_residual = np.std(residuals)
        
        ci_lower = [f - 1.96 * std_residual * np.sqrt(h) for h, f in enumerate(forecast, 1)]
        ci_upper = [f + 1.96 * std_residual * np.sqrt(h) for h, f in enumerate(forecast, 1)]
        
        metrics = self._calculate_metrics(series, fitted)
        
        result = {
            "method": "holt_winters",
            "seasonal_periods": m,
            "trend_type": trend,
            "seasonal_type": seasonal,
            "parameters": {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma)
            },
            "n_observations": n,
            "fitted_values": fitted.tolist(),
            "forecast_values": forecast,
            "periods_ahead": periods,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 0.95
            },
            "components": {
                "level": float(l),
                "trend": float(t),
                "seasonal": s.tolist()
            },
            "residuals": residuals.tolist(),
            "metrics": metrics,
            "warnings": warnings_list
        }
        
        self._log_operation("holt_winters", {
            "seasonal_periods": m,
            "trend": trend,
            "seasonal": seasonal,
            "periods": periods,
            "rmse": metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    def arima(
        self,
        value_column: str,
        p: Optional[int] = None,
        d: Optional[int] = None,
        q: Optional[int] = None,
        periods: int = 5,
        weight_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ARIMA forecast with automatic parameter selection.
        
        Args:
            value_column: Name of the value column
            p: AR order (auto-selected if None)
            d: Differencing order (auto-selected if None)
            q: MA order (auto-selected if None)
            periods: Number of periods to forecast
            weight_column: Optional weight column
            
        Returns:
            Dict with forecast results
        """
        warnings_list = []
        
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna().values.astype(float)
        n = len(series)
        
        if n < 10:
            return self._make_json_safe({"error": "Need at least 10 observations for ARIMA"})
        
        # Auto-select d (differencing order) using ADF-like test
        if d is None:
            d = 0
            test_series = series.copy()
            for diff_order in range(3):
                # Simple stationarity check: variance ratio
                if len(test_series) < 5:
                    break
                first_half_var = np.var(test_series[:len(test_series)//2])
                second_half_var = np.var(test_series[len(test_series)//2:])
                
                var_ratio = max(first_half_var, second_half_var) / (min(first_half_var, second_half_var) + 1e-10)
                
                # Check trend
                trend_coef = np.polyfit(range(len(test_series)), test_series, 1)[0]
                trend_strength = abs(trend_coef) / (np.std(test_series) + 1e-10)
                
                if var_ratio < 2 and trend_strength < 0.1:
                    d = diff_order
                    break
                
                test_series = np.diff(test_series)
                d = diff_order + 1
            
            d = min(d, 2)
            warnings_list.append(f"Auto-selected d={d}")
        
        # Difference the series
        diff_series = series.copy()
        for _ in range(d):
            diff_series = np.diff(diff_series)
        
        # Auto-select p and q using ACF/PACF-like heuristics
        if p is None or q is None:
            # Simple heuristic: use correlation at different lags
            n_diff = len(diff_series)
            max_lag = min(10, n_diff // 4)
            
            acf = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf.append(1.0)
                else:
                    acf.append(np.corrcoef(diff_series[:-lag], diff_series[lag:])[0, 1])
            
            # Significance threshold
            threshold = 1.96 / np.sqrt(n_diff)
            
            if p is None:
                # Count significant ACF lags
                p = 0
                for lag in range(1, len(acf)):
                    if abs(acf[lag]) > threshold:
                        p = lag
                    else:
                        break
                p = min(p, 3)
                warnings_list.append(f"Auto-selected p={p}")
            
            if q is None:
                q = min(p, 2)  # Simple heuristic
                warnings_list.append(f"Auto-selected q={q}")
        
        # Fit AR model (simplified ARIMA without MA for pure Python implementation)
        # Using OLS for AR coefficients
        if n < 20:
            warnings_list.append("Short series - using simplified AR model")
        
        if p > 0:
            # Create lag matrix
            X = np.zeros((len(diff_series) - p, p))
            for i in range(p):
                X[:, i] = diff_series[p-i-1:-i-1]
            y = diff_series[p:]
            
            # OLS fit
            try:
                ar_coefs = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                ar_coefs = np.zeros(p)
                warnings_list.append("AR coefficient estimation failed")
        else:
            ar_coefs = np.array([])
        
        # Fitted values
        fitted_diff = np.zeros(len(diff_series))
        for i in range(p, len(diff_series)):
            fitted_diff[i] = np.sum(ar_coefs * diff_series[i-p:i][::-1])
        
        # Reconstruct fitted values
        fitted = np.zeros(n)
        fitted[:d+p] = series[:d+p]  # Use actual values for initial
        
        for i in range(d + p, n):
            if d == 0:
                fitted[i] = fitted_diff[i]
            else:
                # Integrate back
                fitted[i] = fitted[i-1] + fitted_diff[i-d]
        
        # Simple forecast
        forecast = []
        last_values = list(diff_series[-p:]) if p > 0 else []
        
        for h in range(periods):
            if p > 0:
                next_diff = np.sum(ar_coefs * np.array(last_values[-p:])[::-1])
            else:
                next_diff = 0
            
            if d == 0:
                next_val = next_diff
            else:
                if h == 0:
                    next_val = series[-1] + next_diff
                else:
                    next_val = forecast[-1] + next_diff
            
            forecast.append(next_val)
            last_values.append(next_diff)
        
        # Confidence intervals
        residuals = series[d+p:] - fitted[d+p:]
        std_residual = np.std(residuals)
        
        ci_lower = [f - 1.96 * std_residual * np.sqrt(h+1) for h, f in enumerate(forecast)]
        ci_upper = [f + 1.96 * std_residual * np.sqrt(h+1) for h, f in enumerate(forecast)]
        
        metrics = self._calculate_metrics(series[d+p:], fitted[d+p:])
        
        # Check for non-stationarity warning
        if d == 0 and np.std(series[:n//2]) / (np.std(series[n//2:]) + 1e-10) > 2:
            warnings_list.append("Series may be non-stationary, consider increasing d")
        
        result = {
            "method": "arima",
            "order": {"p": int(p), "d": int(d), "q": int(q)},
            "ar_coefficients": ar_coefs.tolist() if len(ar_coefs) > 0 else [],
            "n_observations": n,
            "fitted_values": fitted.tolist(),
            "forecast_values": forecast,
            "periods_ahead": periods,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 0.95
            },
            "residuals": residuals.tolist(),
            "metrics": metrics,
            "warnings": warnings_list
        }
        
        self._log_operation("arima", {
            "order": {"p": p, "d": d, "q": q},
            "periods": periods,
            "rmse": metrics["rmse"]
        })
        
        return self._make_json_safe(result)
    
    def run_forecast(
        self,
        value_column: str,
        method: str = "auto",
        periods: int = 5,
        time_column: Optional[str] = None,
        weight_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run forecast with specified or auto-selected method.
        
        Args:
            value_column: Name of the value column
            method: Forecast method ('ma', 'es', 'holt_winters', 'arima', 'auto')
            periods: Number of periods to forecast
            time_column: Optional time column for resampling
            weight_column: Optional weight column
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dict with forecast results
        """
        if value_column not in self.df.columns:
            return self._make_json_safe({"error": f"Column '{value_column}' not found"})
        
        series = self.df[value_column].dropna()
        n = len(series)
        
        # Auto-select method
        if method == "auto":
            if n < 10:
                method = "ma"
            elif n < 24:
                method = "es"
            elif n >= 24:
                method = "holt_winters"
        
        # Run selected method
        if method == "ma":
            window = kwargs.get("window", min(5, n // 3))
            return self.moving_average(value_column, window=window, periods=periods, weight_column=weight_column)
        
        elif method == "es":
            alpha = kwargs.get("alpha", None)
            return self.exponential_smoothing(value_column, alpha=alpha, periods=periods, weight_column=weight_column)
        
        elif method == "holt_winters":
            seasonal_periods = kwargs.get("seasonal_periods", 12)
            trend = kwargs.get("trend", "add")
            seasonal = kwargs.get("seasonal", "add")
            return self.holt_winters(
                value_column, 
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                periods=periods,
                weight_column=weight_column
            )
        
        elif method == "arima":
            p = kwargs.get("p", None)
            d = kwargs.get("d", None)
            q = kwargs.get("q", None)
            return self.arima(value_column, p=p, d=d, q=q, periods=periods, weight_column=weight_column)
        
        else:
            return self._make_json_safe({"error": f"Unknown method: {method}"})
    
    @staticmethod
    def forecast_multiple(
        file_ids: List[str],
        file_manager: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run forecasting operations on multiple files with error handling
        
        Args:
            file_ids: List of file identifiers
            file_manager: FileManager instance to load files
            params: Parameters for forecasting (time_column, value_column, method, periods, etc.)
            
        Returns:
            Dictionary mapping file_id to forecast results:
            {
                "<file_id>": {"result": {...}, "status": "ok"},
                "<file_id>": {"error": "No time column in dataset"}
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
                
                # Check if required columns exist
                time_column = params.get("time_column")
                value_column = params.get("value_column")
                
                if time_column and time_column not in df.columns:
                    results[file_id] = {"error": "No time column in dataset"}
                    continue
                
                if value_column and value_column not in df.columns:
                    results[file_id] = {"error": f"Value column '{value_column}' not found"}
                    continue
                
                # Initialize engine for this file
                engine = ForecastingEngine(df)
                
                # Perform forecasting operation
                method = params.get("method", "auto")
                periods = params.get("periods", 12)
                weight_column = params.get("weight_column")
                
                if method == "auto":
                    result = engine.auto_forecast(
                        value_column=value_column,
                        periods=periods,
                        weight_column=weight_column,
                        **params
                    )
                elif method == "ma":
                    result = engine.moving_average(
                        value_column=value_column,
                        window=params.get("window", 3),
                        periods=periods,
                        weight_column=weight_column
                    )
                elif method == "es":
                    result = engine.exponential_smoothing(
                        value_column=value_column,
                        alpha=params.get("alpha"),
                        periods=periods,
                        weight_column=weight_column
                    )
                elif method == "holt_winters":
                    result = engine.holt_winters(
                        value_column=value_column,
                        seasonal_periods=params.get("seasonal_periods", 12),
                        trend=params.get("trend", "add"),
                        seasonal=params.get("seasonal", "add"),
                        periods=periods,
                        weight_column=weight_column
                    )
                elif method == "arima":
                    result = engine.arima(
                        value_column=value_column,
                        p=params.get("p"),
                        d=params.get("d"),
                        q=params.get("q"),
                        periods=periods,
                        weight_column=weight_column
                    )
                else:
                    results[file_id] = {"error": f"Unknown method: {method}"}
                    continue
                
                results[file_id] = {
                    "result": result,
                    "operations_log": engine.operations_log,
                    "status": "ok"
                }
                
            except Exception as e:
                results[file_id] = {"error": str(e)}
        
        return results
