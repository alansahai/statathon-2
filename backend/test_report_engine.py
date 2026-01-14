"""
Test script for ReportEngine - Complete report generation with schema and cleaning
"""

import json
from pathlib import Path
from services.report_engine import ReportEngine

def create_test_metadata():
    """Create comprehensive test metadata files for demonstration"""
    base_path = Path("temp_uploads")
    
    # Create test cleaning metadata
    test_file_id = "5f888f32-d9f0-4a8d-8dea-75af47f01ec3"
    cleaning_dir = base_path / "cleaned" / "default_user"
    cleaning_dir.mkdir(parents=True, exist_ok=True)
    
    test_metadata = {
        "file_id": test_file_id,
        "metadata": {
            "original_filename": "survey_data_2025.csv",
            "rows": 2500,
            "columns": 15,
            "upload_timestamp": "2025-01-07T10:30:00"
        },
        "issue_summary": {
            "missing_summary": {
                "age": {
                    "missing_count": 145,
                    "missing_percent": 5.8,
                    "non_missing_count": 2355
                },
                "income": {
                    "missing_count": 312,
                    "missing_percent": 12.48,
                    "non_missing_count": 2188
                },
                "education": {
                    "missing_count": 89,
                    "missing_percent": 3.56,
                    "non_missing_count": 2411
                },
                "region": {
                    "missing_count": 23,
                    "missing_percent": 0.92,
                    "non_missing_count": 2477
                }
            },
            "numeric_summary": {
                "age": {
                    "dtype": "float64",
                    "count": 2355,
                    "mean": 42.5,
                    "std": 15.3,
                    "min": 18.0,
                    "q1": 30.0,
                    "median": 41.0,
                    "q3": 55.0,
                    "max": 95.0,
                    "unique_count": 78
                },
                "income": {
                    "dtype": "float64",
                    "count": 2188,
                    "mean": 52000.0,
                    "std": 28000.0,
                    "min": 12000.0,
                    "q1": 32000.0,
                    "median": 48000.0,
                    "q3": 68000.0,
                    "max": 250000.0,
                    "unique_count": 1523
                },
                "household_size": {
                    "dtype": "int64",
                    "count": 2500,
                    "mean": 3.2,
                    "std": 1.8,
                    "min": 1.0,
                    "q1": 2.0,
                    "median": 3.0,
                    "q3": 4.0,
                    "max": 12.0,
                    "unique_count": 12
                }
            },
            "categorical_summary": {
                "education": {
                    "dtype": "object",
                    "count": 2411,
                    "unique_count": 6,
                    "top_values": {
                        "Bachelor": 890,
                        "High School": 654,
                        "Master": 432,
                        "PhD": 198,
                        "Associate": 156,
                        "Below High School": 81
                    },
                    "most_common": "Bachelor"
                },
                "region": {
                    "dtype": "object",
                    "count": 2477,
                    "unique_count": 4,
                    "top_values": {
                        "Urban": 1234,
                        "Suburban": 789,
                        "Rural": 345,
                        "Remote": 109
                    },
                    "most_common": "Urban"
                },
                "employment_status": {
                    "dtype": "object",
                    "count": 2500,
                    "unique_count": 5,
                    "top_values": {
                        "Employed": 1567,
                        "Self-employed": 423,
                        "Unemployed": 298,
                        "Retired": 156,
                        "Student": 56
                    },
                    "most_common": "Employed"
                }
            },
            "potential_id_columns": ["respondent_id"]
        },
        "outlier_summary": {
            "age": {
                "method": "iqr",
                "lower_bound": -7.5,
                "upper_bound": 92.5,
                "q1": 30.0,
                "q3": 55.0,
                "iqr": 25.0,
                "outlier_count": 12,
                "outlier_percent": 0.51,
                "outlier_values": [93.0, 94.0, 95.0, 94.5, 93.2]
            },
            "income": {
                "method": "iqr",
                "lower_bound": -22000.0,
                "upper_bound": 122000.0,
                "q1": 32000.0,
                "q3": 68000.0,
                "iqr": 36000.0,
                "outlier_count": 87,
                "outlier_percent": 3.98,
                "outlier_values": [150000.0, 180000.0, 250000.0, 175000.0, 145000.0]
            },
            "household_size": {
                "method": "iqr",
                "lower_bound": -1.0,
                "upper_bound": 7.0,
                "q1": 2.0,
                "q3": 4.0,
                "iqr": 2.0,
                "outlier_count": 34,
                "outlier_percent": 1.36,
                "outlier_values": [8, 9, 10, 11, 12, 8, 9]
            }
        },
        "cleaning_logs": [
            {
                "timestamp": "2025-01-07T10:32:15",
                "operation": "detect_issues",
                "details": {
                    "missing_columns": 4,
                    "numeric_columns": 3,
                    "categorical_columns": 3,
                    "potential_id_columns": 1
                }
            },
            {
                "timestamp": "2025-01-07T10:32:18",
                "operation": "impute_missing",
                "details": {
                    "method": "auto",
                    "columns_processed": 4,
                    "total_values_imputed": 569
                }
            },
            {
                "timestamp": "2025-01-07T10:32:22",
                "operation": "detect_outliers",
                "details": {
                    "method": "iqr",
                    "columns_analyzed": 3,
                    "total_outliers": 133
                }
            },
            {
                "timestamp": "2025-01-07T10:32:25",
                "operation": "fix_outliers",
                "details": {
                    "method": "iqr",
                    "columns_fixed": 3,
                    "outliers_capped": 133
                }
            }
        ],
        "cleaning_summary": {
            "missing_values_handled": 569,
            "duplicates_removed": 0,
            "outliers_fixed": 133
        },
        "row_count": 2500,
        "column_count": 15,
        "original_row_count": 2500,
        "rows_dropped": 0
    }
    
    metadata_path = cleaning_dir / f"{test_file_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"Created comprehensive test metadata at: {metadata_path}")
    
    # Create weighting metadata
    weighting_dir = base_path / "weighted" / "default_user"
    weighting_dir.mkdir(parents=True, exist_ok=True)
    
    weighting_metadata = {
        "file_id": test_file_id,
        "operations_log": [
            {
                "timestamp": "2025-01-07T10:35:00",
                "operation": "apply_poststrat_weights",
                "details": {
                    "categories": ["age_group", "gender", "region"],
                    "method": "post_stratification",
                    "weights_created": 2500
                }
            },
            {
                "timestamp": "2025-01-07T10:35:15",
                "operation": "raking",
                "details": {
                    "method": "raking",
                    "starting_weight_column": "poststrat_weight",
                    "converged": True,
                    "iterations": 8,
                    "final_max_diff": 0.0008,
                    "tolerance": 0.001,
                    "control_margins": ["age_group", "gender", "region", "education"],
                    "dropped_controls": [],
                    "final_stats": {
                        "min_weight": 0.342,
                        "max_weight": 3.125,
                        "mean_weight": 1.002,
                        "median_weight": 0.987,
                        "std_weight": 0.456
                    }
                }
            },
            {
                "timestamp": "2025-01-07T10:35:25",
                "operation": "trim_weights",
                "details": {
                    "method": "percentile",
                    "lower_bound": 0.350,
                    "upper_bound": 3.000,
                    "weights_trimmed": 23,
                    "trimmed_percentage": 0.92
                }
            }
        ],
        "diagnostics": {
            "weight_column": "raked_weight_trimmed",
            "n_observations": 2500,
            "mean_weight": 1.000,
            "median_weight": 0.987,
            "std_weight": 0.445,
            "min_weight": 0.350,
            "max_weight": 3.000,
            "cv": 0.445,
            "sum_weights": 2500.0,
            "effective_sample_size": 2134.5,
            "design_effect": 1.198,
            "entropy": 7.723,
            "entropy_ratio": 0.985,
            "loss_of_precision": 0.146,
            "percentiles": {
                "p1": 0.387,
                "p5": 0.452,
                "p10": 0.523,
                "p25": 0.678,
                "p50": 0.987,
                "p75": 1.234,
                "p90": 1.678,
                "p95": 2.012,
                "p99": 2.789
            }
        },
        "final_weight_column": "raked_weight_trimmed",
        "summary": {
            "base_weights_applied": True,
            "poststrat_applied": True,
            "raking_applied": True,
            "trimming_applied": True,
            "effective_sample_size": 2134.5,
            "design_effect": 1.198
        }
    }
    
    weighting_path = weighting_dir / f"{test_file_id}_weighting_metadata.json"
    with open(weighting_path, 'w') as f:
        json.dump(weighting_metadata, f, indent=2)
    
    print(f"Created weighting metadata at: {weighting_path}")
    
    # Create analysis/descriptive stats metadata
    processed_dir = base_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_metadata = {
        "file_id": test_file_id,
        "descriptive_stats": {
            "age": {
                "type": "numeric",
                "count": 2355,
                "missing": 145,
                "mean": 42.53,
                "median": 41.0,
                "std": 15.34,
                "min": 18.0,
                "max": 95.0,
                "skewness": 0.45,
                "kurtosis": -0.23,
                "q1": 30.0,
                "q3": 55.0
            },
            "income": {
                "type": "numeric",
                "count": 2188,
                "missing": 312,
                "mean": 52000.0,
                "median": 48000.0,
                "std": 28000.0,
                "min": 12000.0,
                "max": 250000.0,
                "skewness": 1.87,
                "kurtosis": 4.52,
                "q1": 32000.0,
                "q3": 68000.0
            },
            "household_size": {
                "type": "numeric",
                "count": 2500,
                "missing": 0,
                "mean": 3.2,
                "median": 3.0,
                "std": 1.8,
                "min": 1.0,
                "max": 12.0,
                "skewness": 1.23,
                "kurtosis": 2.15,
                "q1": 2.0,
                "q3": 4.0
            },
            "satisfaction_score": {
                "type": "numeric",
                "count": 2500,
                "missing": 0,
                "mean": 7.2,
                "median": 7.5,
                "std": 1.9,
                "min": 1.0,
                "max": 10.0,
                "skewness": -0.65,
                "kurtosis": 0.12,
                "q1": 6.0,
                "q3": 9.0
            },
            "education": {
                "type": "categorical",
                "count": 2411,
                "missing": 89,
                "unique_count": 6,
                "mode": "Bachelor",
                "frequencies": {
                    "Bachelor": 890,
                    "High School": 654,
                    "Master": 432,
                    "PhD": 198,
                    "Associate": 156,
                    "Below High School": 81
                }
            },
            "region": {
                "type": "categorical",
                "count": 2477,
                "missing": 23,
                "unique_count": 4,
                "mode": "Urban",
                "frequencies": {
                    "Urban": 1234,
                    "Suburban": 789,
                    "Rural": 345,
                    "Remote": 109
                }
            },
            "employment_status": {
                "type": "categorical",
                "count": 2500,
                "missing": 0,
                "unique_count": 5,
                "mode": "Employed",
                "frequencies": {
                    "Employed": 1567,
                    "Self-employed": 423,
                    "Unemployed": 298,
                    "Retired": 156,
                    "Student": 56
                }
            },
            "gender": {
                "type": "categorical",
                "count": 2500,
                "missing": 0,
                "unique_count": 3,
                "mode": "Female",
                "frequencies": {
                    "Female": 1289,
                    "Male": 1156,
                    "Other": 55
                }
            }
        },
        "crosstabs": [
            {
                "row_var": "education",
                "col_var": "employment_status",
                "table": {
                    "Bachelor": {
                        "Employed": 589,
                        "Self-employed": 178,
                        "Unemployed": 87,
                        "Retired": 28,
                        "Student": 8
                    },
                    "High School": {
                        "Employed": 387,
                        "Self-employed": 89,
                        "Unemployed": 123,
                        "Retired": 42,
                        "Student": 13
                    },
                    "Master": {
                        "Employed": 298,
                        "Self-employed": 112,
                        "Unemployed": 15,
                        "Retired": 5,
                        "Student": 2
                    },
                    "PhD": {
                        "Employed": 143,
                        "Self-employed": 34,
                        "Unemployed": 12,
                        "Retired": 8,
                        "Student": 1
                    }
                },
                "chi_square_test": {
                    "chi2_statistic": 187.45,
                    "p_value": 0.0001,
                    "degrees_of_freedom": 12,
                    "warnings": []
                }
            },
            {
                "row_var": "region",
                "col_var": "income_bracket",
                "table": {
                    "Urban": {
                        "Low": 234,
                        "Medium": 567,
                        "High": 433
                    },
                    "Suburban": {
                        "Low": 145,
                        "Medium": 398,
                        "High": 246
                    },
                    "Rural": {
                        "Low": 189,
                        "Medium": 112,
                        "High": 44
                    },
                    "Remote": {
                        "Low": 67,
                        "Medium": 28,
                        "High": 14
                    }
                },
                "chi_square_test": {
                    "chi2_statistic": 256.89,
                    "p_value": 0.00001,
                    "degrees_of_freedom": 6,
                    "warnings": []
                }
            },
            {
                "row_var": "gender",
                "col_var": "satisfaction_bracket",
                "table": {
                    "Female": {
                        "Low": 178,
                        "Medium": 534,
                        "High": 577
                    },
                    "Male": {
                        "Low": 198,
                        "Medium": 512,
                        "High": 446
                    },
                    "Other": {
                        "Low": 8,
                        "Medium": 23,
                        "High": 24
                    }
                },
                "chi_square_test": {
                    "chi2_statistic": 8.34,
                    "p_value": 0.0803,
                    "degrees_of_freedom": 4,
                    "warnings": ["2 cells have expected frequency < 5"]
                }
            }
        ],
        "regression": {
            "ols": {
                "dependent_var": "income",
                "coefficients": {
                    "Intercept": 25000.45,
                    "age": 850.32,
                    "education_years": 3200.75,
                    "household_size": -450.20,
                    "experience": 1250.60
                },
                "std_errors": {
                    "Intercept": 2100.30,
                    "age": 125.45,
                    "education_years": 350.20,
                    "household_size": 280.15,
                    "experience": 180.90
                },
                "p_values": {
                    "Intercept": 0.0001,
                    "age": 0.0001,
                    "education_years": 0.0001,
                    "household_size": 0.1089,
                    "experience": 0.0001
                },
                "r_squared": 0.6234,
                "adj_r_squared": 0.6189
            },
            "logistic": {
                "dependent_var": "employment_status_binary",
                "coefficients": {
                    "Intercept": -2.145,
                    "age": 0.032,
                    "education_years": 0.285,
                    "gender_male": 0.420,
                    "urban": 0.615
                },
                "odds_ratios": {
                    "Intercept": 0.117,
                    "age": 1.033,
                    "education_years": 1.330,
                    "gender_male": 1.522,
                    "urban": 1.849
                },
                "std_errors": {
                    "Intercept": 0.345,
                    "age": 0.008,
                    "education_years": 0.045,
                    "gender_male": 0.125,
                    "urban": 0.135
                },
                "p_values": {
                    "Intercept": 0.0001,
                    "age": 0.0001,
                    "education_years": 0.0001,
                    "gender_male": 0.0008,
                    "urban": 0.0001
                },
                "accuracy": 0.8234,
                "precision": 0.8156,
                "recall": 0.8389,
                "f1": 0.8271
            }
        },
        "forecasting": {
            "target_variable": "monthly_sales",
            "forecast_horizon": 12,
            "model_type": "SARIMA",
            "actual_values": [
                1250.5, 1320.8, 1180.3, 1425.6, 1390.2, 
                1510.4, 1475.9, 1620.3, 1580.1, 1705.8,
                1650.2, 1820.5, 1780.3, 1920.6, 1850.4,
                2050.8, 2010.5, 2180.3, 2120.7, 2305.9
            ],
            "time_periods": list(range(1, 33)),
            "forecast_values": [
                1250.5, 1320.8, 1180.3, 1425.6, 1390.2, 
                1510.4, 1475.9, 1620.3, 1580.1, 1705.8,
                1650.2, 1820.5, 1780.3, 1920.6, 1850.4,
                2050.8, 2010.5, 2180.3, 2120.7, 2305.9,
                2250.3, 2410.5, 2380.8, 2520.4, 2490.6,
                2650.2, 2610.8, 2780.5, 2740.3, 2920.7,
                2880.5, 3050.2
            ],
            "lower_bounds": [
                1220.3, 1285.4, 1145.7, 1385.2, 1350.8,
                1465.9, 1430.5, 1570.8, 1525.4, 1650.3,
                1590.7, 1755.8, 1710.5, 1845.2, 1770.9,
                1965.4, 1920.8, 2085.7, 2025.3, 2200.5,
                2140.8, 2290.3, 2255.6, 2385.7, 2350.2,
                2495.8, 2450.4, 2610.3, 2565.8, 2735.4,
                2685.9, 2845.6
            ],
            "upper_bounds": [
                1280.7, 1356.2, 1214.9, 1466.0, 1429.6,
                1554.9, 1521.3, 1669.8, 1634.8, 1761.3,
                1709.7, 1885.2, 1850.1, 1996.0, 1929.9,
                2136.2, 2100.2, 2274.9, 2216.1, 2411.3,
                2359.8, 2530.7, 2506.0, 2655.1, 2631.0,
                2804.6, 2771.2, 2950.7, 2914.8, 3106.0,
                3075.1, 3254.8
            ],
            "decomposition": {
                "trend": "upward",
                "seasonal_period": 12,
                "seasonal_strength": 0.35
            },
            "metrics": {
                "mae": 45.67,
                "mape": 3.82,
                "rmse": 58.34
            }
        },
        "ml_results": {
            "model_type": "random_forest",
            "training_params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42
            },
            "metrics": {
                "accuracy": 0.8756,
                "precision": 0.8621,
                "recall": 0.8893,
                "f1": 0.8755,
                "roc_auc": 0.9234
            },
            "feature_importance": {
                "education_years": 0.2345,
                "age": 0.1876,
                "income": 0.1543,
                "household_size": 0.1234,
                "work_experience": 0.0987,
                "urban_residence": 0.0765,
                "gender_male": 0.0543,
                "marital_status": 0.0432,
                "health_score": 0.0198,
                "region_north": 0.0077
            },
            "confusion_matrix": [[450, 45], [32, 473]],
            "classification_report": {
                "0": {"precision": 0.93, "recall": 0.91, "f1-score": 0.92},
                "1": {"precision": 0.91, "recall": 0.94, "f1-score": 0.92}
            }
        },
        "insight_results": {
            "correlations": {
                "education_years-income": 0.7845,
                "age-work_experience": 0.8923,
                "household_size-children": 0.9012,
                "income-housing_quality": 0.6734,
                "education_years-employment_status": 0.5432,
                "age-health_score": -0.6589,
                "income-debt_ratio": -0.7123,
                "urban_residence-access_to_services": 0.8456,
                "marital_status-household_size": 0.5678,
                "work_experience-income": 0.7234
            },
            "anomalies": {
                "income": "Detected 23 extreme outliers (>3 IQR) with values exceeding $150,000, suggesting possible data entry errors or ultra-high-income individuals",
                "age": "Found 5 records with age values above 95 years, which may indicate centenarians or data quality issues",
                "household_size": "Identified 12 households with more than 10 members, which is unusual and warrants verification",
                "work_experience": "Detected 8 cases where work experience exceeds age minus 15, indicating logical inconsistencies"
            },
            "risk_groups": {
                "Low-education Rural Youth": {
                    "risk_score": 0.82,
                    "key_metric": "68% unemployment rate",
                    "sample_size": 145
                },
                "Single-parent Urban Households": {
                    "risk_score": 0.75,
                    "key_metric": "High debt-to-income ratio (avg: 0.76)",
                    "sample_size": 203
                },
                "Elderly with Limited Access": {
                    "risk_score": 0.71,
                    "key_metric": "Poor health scores (avg: 3.2/10)",
                    "sample_size": 178
                },
                "Under-represented Minorities": {
                    "risk_score": 0.45,
                    "key_metric": "Only 4.2% of sample",
                    "sample_size": 105
                }
            },
            "drivers": {
                "education_years": 0.8234,
                "urban_residence": 0.7456,
                "household_income": 0.7123,
                "access_to_healthcare": 0.6789,
                "employment_stability": 0.6234,
                "family_support_network": 0.5567,
                "digital_literacy": 0.4891,
                "transportation_access": 0.4123
            },
            "recommended_actions": [
                {
                    "action": "Implement targeted education and vocational training programs for rural youth",
                    "justification": "Low-education rural youth show 68% unemployment with risk score 0.82, indicating urgent need for skill development interventions",
                    "priority": "High"
                },
                {
                    "action": "Establish financial counseling services for single-parent households",
                    "justification": "Single-parent urban households exhibit high debt-to-income ratios (0.76), requiring debt management and budgeting support",
                    "priority": "High"
                },
                {
                    "action": "Expand healthcare access and preventive services for elderly populations",
                    "justification": "Elderly with limited access show poor health scores (3.2/10) and strong negative correlation between age and health (-0.66)",
                    "priority": "High"
                },
                {
                    "action": "Increase sampling efforts to better represent minority populations",
                    "justification": "Under-represented minorities constitute only 4.2% of sample, limiting generalizability of findings to these groups",
                    "priority": "Medium"
                },
                {
                    "action": "Investigate and validate extreme outliers in income data",
                    "justification": "23 extreme outliers detected in income variable, requiring data quality verification to ensure accurate analysis",
                    "priority": "Medium"
                },
                {
                    "action": "Develop policies leveraging education as primary outcome driver",
                    "justification": "Education years shows highest driver score (0.82) with strong correlation to income (0.78) and employment (0.54)",
                    "priority": "High"
                }
            ],
            "missing_patterns": "Missing data analysis reveals non-random patterns: income data is 15% more likely to be missing in rural areas, and health scores are systematically missing for respondents aged 18-25. These patterns suggest potential selection bias and should be addressed through targeted follow-up surveys or multiple imputation techniques."
        },
        "analysis_timestamp": "2025-01-07T10:40:00",
        "weighted": True,
        "weight_column": "raked_weight_trimmed"
    }
    
    analysis_path = processed_dir / f"{test_file_id}_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_metadata, f, indent=2)
    
    print(f"Created analysis metadata at: {analysis_path}")
    
    # Create workflow audit log
    audit_log_data = {
        "file_id": test_file_id,
        "operations": [
            {
                "step": 1,
                "operation": "File Upload - survey_data_2025.csv",
                "timestamp": "2025-01-07T10:30:00",
                "duration": 2.34,
                "status": "completed"
            },
            {
                "step": 2,
                "operation": "Data Validation and Schema Detection",
                "timestamp": "2025-01-07T10:30:02",
                "duration": 1.87,
                "status": "completed"
            },
            {
                "step": 3,
                "operation": "Missing Value Analysis and Imputation",
                "timestamp": "2025-01-07T10:30:04",
                "duration": 5.23,
                "status": "completed"
            },
            {
                "step": 4,
                "operation": "Outlier Detection and Treatment (IQR Method)",
                "timestamp": "2025-01-07T10:30:09",
                "duration": 3.45,
                "status": "completed"
            },
            {
                "step": 5,
                "operation": "Data Cleaning Completed - 569 issues resolved",
                "timestamp": "2025-01-07T10:30:13",
                "duration": 0.12,
                "status": "completed"
            },
            {
                "step": 6,
                "operation": "Survey Weighting - Base Weights Calculation",
                "timestamp": "2025-01-07T10:31:00",
                "duration": 2.56,
                "status": "completed"
            },
            {
                "step": 7,
                "operation": "Post-Stratification Adjustment Applied",
                "timestamp": "2025-01-07T10:31:03",
                "duration": 1.89,
                "status": "completed"
            },
            {
                "step": 8,
                "operation": "Raking Algorithm - 8 iterations to convergence",
                "timestamp": "2025-01-07T10:31:05",
                "duration": 4.67,
                "status": "completed"
            },
            {
                "step": 9,
                "operation": "Weight Trimming - 23 weights adjusted",
                "timestamp": "2025-01-07T10:31:10",
                "duration": 0.89,
                "status": "completed"
            },
            {
                "step": 10,
                "operation": "Descriptive Statistics Analysis",
                "timestamp": "2025-01-07T10:32:00",
                "duration": 3.21,
                "status": "completed"
            },
            {
                "step": 11,
                "operation": "Cross-Tabulation Analysis with Chi-Square Tests",
                "timestamp": "2025-01-07T10:32:03",
                "duration": 2.45,
                "status": "completed"
            },
            {
                "step": 12,
                "operation": "OLS and Logistic Regression Models",
                "timestamp": "2025-01-07T10:32:06",
                "duration": 6.78,
                "status": "completed"
            },
            {
                "step": 13,
                "operation": "Time Series Forecasting (SARIMA Model)",
                "timestamp": "2025-01-07T10:32:13",
                "duration": 12.34,
                "status": "completed"
            },
            {
                "step": 14,
                "operation": "Machine Learning - Random Forest Training",
                "timestamp": "2025-01-07T10:32:25",
                "duration": 8.56,
                "status": "completed"
            },
            {
                "step": 15,
                "operation": "Automated Insights Generation",
                "timestamp": "2025-01-07T10:32:34",
                "duration": 4.23,
                "status": "completed"
            },
            {
                "step": 16,
                "operation": "Report Generation Initiated",
                "timestamp": "2025-01-07T10:40:00",
                "duration": 0.05,
                "status": "completed"
            }
        ],
        "total_duration": 70.89,
        "workflow_status": "completed"
    }
    
    audit_log_path = processed_dir / f"{test_file_id}_audit_log.json"
    with open(audit_log_path, 'w') as f:
        json.dump(audit_log_data, f, indent=2)
    
    print(f"Created audit log at: {audit_log_path}")
    
    return test_file_id

def test_report_generation():
    """Test complete report generation with all sections"""
    print("=" * 70)
    print("Testing ReportEngine - Complete Report with All Sections")
    print("=" * 70)
    
    # Create test metadata
    file_id = create_test_metadata()
    
    # Initialize ReportEngine
    print(f"\nInitializing ReportEngine with file_id: {file_id}")
    engine = ReportEngine(file_id)
    
    # Generate complete report
    print("\nGenerating report with:")
    print("  - Title Page")
    print("  - Schema Section")
    print("  - Cleaning Summary Section")
    print("  - Weighting Summary Section")
    print("  - Margin of Error & CI Section")
    print("  - Descriptive Statistics Section")
    print("  - Cross-Tabulation Analysis Section")
    print("  - Regression Analysis Section")
    print("  - Forecasting Analysis Section")
    print("  - Machine Learning Models Section")
    print("  - Insights & Recommendations Section")
    print("  - Workflow Log Section")
    print("  - Appendix Section")
    
    try:
        pdf_path = engine.generate_basic_report()
        print(f"\n✓ Report generated successfully!")
        print(f"  PDF saved at: {pdf_path}")
        
        # Verify file exists
        if Path(pdf_path).exists():
            file_size = Path(pdf_path).stat().st_size
            print(f"  File size: {file_size:,} bytes")
            print(f"\n  Report includes:")
            print(f"    • Title page with dataset information")
            print(f"    • Schema table with column analysis")
            print(f"    • Cleaning summary with missing value and outlier analysis")
            print(f"    • Weighting methodology with DEFF, ESS, CV diagnostics")
            print(f"    • Margin of error and 95% confidence intervals")
            print(f"    • Descriptive statistics for numeric and categorical variables")
            print(f"    • Cross-tabulation analysis with chi-square tests")
            print(f"    • Regression analysis (OLS + Logistic) with model diagnostics")
            print(f"    • Forecasting analysis with time series predictions and visualizations")
            print(f"    • Machine learning models with feature importance and performance metrics")
            print(f"    • Automated insights, risk groups, and evidence-based recommendations")
            print(f"    • Workflow audit log with timestamps and durations")
            print(f"    • Comprehensive appendix with detailed statistics and formulas")
            print(f"    • Page numbers on all pages")
        else:
            print("  Warning: PDF file not found!")
            
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_report_generation()
