# Weighting Engine API Examples

Complete examples for using the StatFlow AI Survey Weighting API.

---

## Table of Contents
1. [Base Weights](#1-base-weights)
2. [Post-Stratification](#2-post-stratification)
3. [Raking (Iterative Proportional Fitting)](#3-raking)
4. [Weight Validation](#4-weight-validation)
5. [Weight Trimming](#5-weight-trimming)
6. [Diagnostics](#6-diagnostics)

---

## 1. Base Weights

Calculate base weights from inclusion probabilities (inverse probability weighting).

### Endpoint
```
POST /api/weighting/calculate
```

### Request Body
```json
{
  "file_id": "abc123def456",
  "method": "base",
  "inclusion_prob_column": "selection_probability"
}
```

### Response
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "weighted_file_path": "temp_uploads/weighted/default_user/abc123def456_weighted.csv",
  "method": "base",
  "result": {
    "method": "base",
    "log_entry": {
      "timestamp": "2026-01-07T10:30:00.000000",
      "operation": "calculate_base_weights",
      "details": {
        "inclusion_prob_column": "selection_probability",
        "summary": {
          "min_weight": 1.25,
          "max_weight": 8.33,
          "mean_weight": 3.45,
          "median_weight": 3.20,
          "std_weight": 1.82,
          "total_weight": 3450.0
        }
      }
    },
    "summary": {
      "min_weight": 1.25,
      "max_weight": 8.33,
      "mean_weight": 3.45,
      "median_weight": 3.20,
      "std_weight": 1.82,
      "total_weight": 3450.0
    }
  },
  "operations_log": [
    {
      "timestamp": "2026-01-07T10:30:00.000000",
      "operation": "initialization",
      "details": {
        "rows": 1000,
        "columns": 15,
        "numeric_cols": 8,
        "categorical_cols": 7
      }
    },
    {
      "timestamp": "2026-01-07T10:30:01.000000",
      "operation": "calculate_base_weights",
      "details": {
        "inclusion_prob_column": "selection_probability",
        "summary": {
          "min_weight": 1.25,
          "max_weight": 8.33,
          "mean_weight": 3.45,
          "median_weight": 3.20,
          "std_weight": 1.82,
          "total_weight": 3450.0
        }
      }
    }
  ]
}
```

### Example CSV Data Format
Your uploaded CSV should contain:
```csv
respondent_id,age,gender,selection_probability,response
1,25,Male,0.15,Yes
2,35,Female,0.20,Yes
3,45,Male,0.12,No
```

---

## 2. Post-Stratification

Adjust weights to match known population distributions for specific strata.

### Endpoint
```
POST /api/weighting/calculate
```

### Request Body
```json
{
  "file_id": "abc123def456",
  "method": "poststrat",
  "strata_column": "age_group",
  "population_totals": {
    "18-24": 5000,
    "25-34": 8000,
    "35-44": 7500,
    "45-54": 6000,
    "55+": 4500
  }
}
```

### Response
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "weighted_file_path": "temp_uploads/weighted/default_user/abc123def456_weighted.csv",
  "method": "poststrat",
  "result": {
    "method": "poststrat",
    "log_entry": {
      "timestamp": "2026-01-07T10:35:00.000000",
      "operation": "apply_poststrat_weights",
      "details": {
        "strata_column": "age_group",
        "num_strata": 5,
        "adjustment_factors": {
          "18-24": 1.25,
          "25-34": 0.89,
          "35-44": 1.05,
          "45-54": 1.15,
          "55+": 0.98
        },
        "sample_totals": {
          "18-24": 4000,
          "25-34": 9000,
          "35-44": 7143,
          "45-54": 5217,
          "55+": 4592
        },
        "population_totals": {
          "18-24": 5000,
          "25-34": 8000,
          "35-44": 7500,
          "45-54": 6000,
          "55+": 4500
        },
        "final_weight_sum": 31000.0,
        "mean_weight": 3.87,
        "std_weight": 2.15
      }
    },
    "summary": {
      "strata_column": "age_group",
      "num_strata": 5,
      "adjustment_factors": {
        "18-24": 1.25,
        "25-34": 0.89,
        "35-44": 1.05,
        "45-54": 1.15,
        "55+": 0.98
      },
      "final_weight_sum": 31000.0,
      "mean_weight": 3.87,
      "std_weight": 2.15
    }
  }
}
```

**Note:** Must run `calculate_base_weights()` first, or the engine will require base_weight column in your data.

---

## 3. Raking (Iterative Proportional Fitting)

Adjust weights to match population distributions across multiple variables simultaneously.

### Endpoint
```
POST /api/weighting/calculate
```

### Request Body
```json
{
  "file_id": "abc123def456",
  "method": "raking",
  "control_totals": {
    "gender": {
      "Male": 0.48,
      "Female": 0.52
    },
    "education": {
      "High School": 0.25,
      "Bachelor": 0.45,
      "Graduate": 0.30
    },
    "region": {
      "North": 0.20,
      "South": 0.35,
      "East": 0.25,
      "West": 0.20
    }
  },
  "max_iterations": 50,
  "tolerance": 0.001
}
```

### Response
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "weighted_file_path": "temp_uploads/weighted/default_user/abc123def456_weighted.csv",
  "method": "raking",
  "result": {
    "method": "raking",
    "log_entry": {
      "timestamp": "2026-01-07T10:40:00.000000",
      "operation": "raking",
      "details": {
        "converged": true,
        "iterations": 12,
        "final_max_diff": 0.0008,
        "tolerance": 0.001,
        "control_margins": ["gender", "education", "region"],
        "iteration_log": [
          {
            "iteration": 1,
            "max_weight_diff": 0.156,
            "mean_weight": 3.52,
            "total_weight": 3520.0
          },
          {
            "iteration": 2,
            "max_weight_diff": 0.089,
            "mean_weight": 3.51,
            "total_weight": 3510.0
          },
          {
            "iteration": 12,
            "max_weight_diff": 0.0008,
            "mean_weight": 3.50,
            "total_weight": 3500.0
          }
        ],
        "final_stats": {
          "min_weight": 0.85,
          "max_weight": 9.23,
          "mean_weight": 3.50,
          "median_weight": 3.15,
          "std_weight": 2.10
        }
      }
    },
    "summary": {
      "converged": true,
      "iterations": 12,
      "final_max_diff": 0.0008,
      "tolerance": 0.001
    }
  }
}
```

**Important:** Control totals must be proportions that sum to 1.0 for each variable.

---

## 4. Weight Validation

Validate that calculated weights meet quality standards.

### Endpoint
```
POST /api/weighting/validate
```

### Request Body
```json
{
  "file_id": "abc123def456",
  "weight_column": "raked_weight"
}
```

If `weight_column` is omitted, the engine auto-detects the most recent weight column.

### Response (Passing)
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "validation": {
    "status": "pass",
    "weight_column": "raked_weight",
    "problems": [],
    "n_observations": 1000,
    "n_valid": 1000,
    "statistics": {
      "min": 0.85,
      "max": 9.23,
      "mean": 3.50,
      "median": 3.15,
      "std": 2.10
    }
  }
}
```

### Response (With Problems)
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "validation": {
    "status": "fail",
    "weight_column": "raked_weight",
    "problems": [
      "Contains 5 NaN values",
      "Contains 2 zero values",
      "Contains 1 negative values"
    ],
    "n_observations": 1000,
    "n_valid": 992,
    "statistics": {
      "min": -0.15,
      "max": 9.23,
      "mean": 3.42,
      "median": 3.10,
      "std": 2.25
    }
  }
}
```

---

## 5. Weight Trimming

Cap extreme weights to reduce variance while maintaining approximate balance.

### Endpoint
```
POST /api/weighting/trim
```

### Request Body
```json
{
  "file_id": "abc123def456",
  "min_w": 0.3,
  "max_w": 3.0,
  "weight_column": "raked_weight"
}
```

Default values: `min_w=0.3`, `max_w=3.0`. If `weight_column` is omitted, auto-detects.

### Response
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "trimmed_file_path": "temp_uploads/weighted/default_user/abc123def456_weighted_trimmed.csv",
  "summary": {
    "weight_column": "raked_weight",
    "min_threshold": 0.3,
    "max_threshold": 3.0,
    "n_below_min": 23,
    "n_above_max": 47,
    "total_trimmed": 70,
    "pct_trimmed": 7.0,
    "original_stats": {
      "min": 0.12,
      "max": 9.23,
      "mean": 3.50,
      "std": 2.10
    },
    "trimmed_stats": {
      "min": 0.3,
      "max": 3.0,
      "mean": 2.95,
      "std": 1.45
    }
  },
  "operations_log": [
    {
      "timestamp": "2026-01-07T10:45:00.000000",
      "operation": "trim_weights",
      "details": {
        "weight_column": "raked_weight",
        "min_threshold": 0.3,
        "max_threshold": 3.0,
        "total_trimmed": 70,
        "pct_trimmed": 7.0
      }
    }
  ]
}
```

---

## 6. Diagnostics

Get comprehensive weight diagnostics including effective sample size and design effects.

### Endpoint
```
GET /api/weighting/diagnostics/{file_id}?weight_column=raked_weight_trimmed
```

Query parameter `weight_column` is optional (auto-detects if omitted).

### Response
```json
{
  "status": "success",
  "file_id": "abc123def456",
  "diagnostics": {
    "weight_column": "raked_weight_trimmed",
    "n_observations": 1000,
    "mean_weight": 2.95,
    "median_weight": 2.85,
    "std_weight": 1.45,
    "min_weight": 0.3,
    "max_weight": 3.0,
    "cv": 0.492,
    "sum_weights": 2950.0,
    "effective_sample_size": 758.5,
    "design_effect": 1.242,
    "percentiles": {
      "p1": 0.42,
      "p5": 0.68,
      "p10": 0.95,
      "p25": 1.85,
      "p50": 2.85,
      "p75": 2.98,
      "p90": 2.99,
      "p95": 3.0,
      "p99": 3.0
    },
    "distribution": {
      "skewness": -0.125,
      "kurtosis": 2.45
    }
  }
}
```

### Key Metrics Explained

- **cv (Coefficient of Variation):** std / mean. Lower is better. Values < 0.5 are good.
- **effective_sample_size:** Adjusted sample size accounting for weight variability. Formula: (Σw)² / Σw²
- **design_effect:** 1 + cv². Measures increase in variance due to weighting. Values close to 1 are ideal.
- **percentiles:** Distribution of weights across the sample.

---

## Workflow Examples

### Complete Workflow 1: Base Weights → Trimming → Diagnostics

```bash
# Step 1: Calculate base weights
curl -X POST http://localhost:8000/api/weighting/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "method": "base",
    "inclusion_prob_column": "prob"
  }'

# Step 2: Validate weights
curl -X POST http://localhost:8000/api/weighting/validate \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "weight_column": "base_weight"
  }'

# Step 3: Trim extreme weights
curl -X POST http://localhost:8000/api/weighting/trim \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "min_w": 0.5,
    "max_w": 2.5
  }'

# Step 4: Get final diagnostics
curl -X GET "http://localhost:8000/api/weighting/diagnostics/abc123?weight_column=base_weight_trimmed"
```

### Complete Workflow 2: Base → Post-Strat → Raking → Trim

```bash
# Step 1: Base weights
POST /api/weighting/calculate
{
  "file_id": "survey2025",
  "method": "base",
  "inclusion_prob_column": "selection_prob"
}

# Step 2: Post-stratification (requires base_weight from Step 1)
POST /api/weighting/calculate
{
  "file_id": "survey2025",
  "method": "poststrat",
  "strata_column": "age_group",
  "population_totals": {
    "18-24": 1200,
    "25-44": 3500,
    "45-64": 2800,
    "65+": 1500
  }
}

# Step 3: Raking for fine-tuning
POST /api/weighting/calculate
{
  "file_id": "survey2025",
  "method": "raking",
  "control_totals": {
    "gender": {"Male": 0.49, "Female": 0.51},
    "education": {"HS": 0.30, "College": 0.45, "Grad": 0.25}
  },
  "max_iterations": 50,
  "tolerance": 0.001
}

# Step 4: Trim final weights
POST /api/weighting/trim
{
  "file_id": "survey2025",
  "min_w": 0.4,
  "max_w": 3.5
}

# Step 5: Final diagnostics
GET /api/weighting/diagnostics/survey2025
```

---

## Error Handling

### Common Errors

**404 - File Not Found**
```json
{
  "detail": "File not found"
}
```
Solution: Ensure file_id is correct and file was uploaded via `/api/upload/upload`.

**400 - Missing Required Parameters**
```json
{
  "detail": "inclusion_prob_column required for base weights"
}
```
Solution: Include all required parameters for the chosen method.

**400 - Invalid Control Totals**
```json
{
  "detail": "Control totals for 'gender' sum to 0.95, expected ~1.0"
}
```
Solution: Ensure proportions sum to 1.0 (within 0.01 tolerance).

**400 - Column Not Found**
```json
{
  "detail": "Column 'selection_prob' not found in DataFrame"
}
```
Solution: Check column names in your uploaded CSV match the API request.

---

## Notes

1. **JSON-Safe Output:** All responses automatically convert NaN/Inf to `null`, ensuring valid JSON.

2. **Operations Log:** Each weighting operation is logged with timestamps. Access via `operations_log` in response.

3. **File Persistence:** Weighted files are saved to `temp_uploads/weighted/default_user/` directory.

4. **Auto-Detection:** Weight column auto-detection follows priority: `raked_weight_trimmed` → `raked_weight` → `poststrat_weight` → `base_weight`.

5. **Raking Convergence:** If raking doesn't converge within `max_iterations`, results are still returned but `converged: false`.

6. **Memory Management:** Large files (>100MB) may require increased server memory.

---

## Testing with Swagger UI

Access interactive API documentation at:
```
http://localhost:8000/docs
```

Navigate to "Weighting" section to test all endpoints directly in browser.
