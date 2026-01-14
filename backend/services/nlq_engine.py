"""
Natural Language Query Engine (NLQEngine)
Rule-based query parsing and routing for StatFlow AI

Parses natural language queries and routes them to appropriate analysis engines.
"""

from typing import Dict, List, Optional
import re
import json
from pathlib import Path
from datetime import datetime


class NLQEngine:
    """
    Natural Language Query Engine for intelligent query parsing and routing.
    
    Uses rule-based pattern matching to:
    1. Detect user intent from query keywords
    2. Extract relevant column names from the query
    3. Route to appropriate analysis engine with parameters
    """
    
    def __init__(self, dataframe):
        """
        Initialize NLQEngine with dataset.
        
        Args:
            dataframe: Pandas DataFrame to analyze
        """
        self.df = dataframe
        self.columns = dataframe.columns.tolist()
        
        # Preprocess column names for matching (lowercase, no spaces)
        self.column_map = {
            col.lower().replace('_', ' ').replace('-', ' '): col 
            for col in self.columns
        }
        
        # Intent keyword patterns
        self.intent_patterns = {
            'compare': [
                'compare', 'comparison', 'difference', 'between', 
                'vs', 'versus', 'contrast', 'group by', 'by group',
                'across', 'male and female', 'by gender', 'by category'
            ],
            'trend': [
                'trend', 'over time', 'time series', 'timeseries',
                'forecast', 'projection', 'historical', 'progression',
                'evolution', 'change over', 'pattern over'
            ],
            'distribution': [
                'distribution', 'histogram', 'frequency', 'spread',
                'range', 'variance', 'how distributed', 'breakdown',
                'statistics', 'descriptive'
            ],
            'relationship': [
                'relationship', 'correlation', 'correlate', 'associated',
                'association', 'relate', 'connection', 'linked',
                'depends on', 'effect of', 'impact of'
            ],
            'predict': [
                'predict', 'prediction', 'forecast', 'model',
                'estimate', 'classify', 'classification', 'regression',
                'machine learning', 'ml model'
            ],
            'risk': [
                'risk', 'risk group', 'vulnerable', 'high risk',
                'at risk', 'danger', 'outlier', 'anomaly',
                'abnormal', 'flagged'
            ],
            'summary': [
                'summary', 'summarize', 'overview', 'describe',
                'describe dataset', 'what is', 'tell me about',
                'general', 'overall'
            ]
        }
        
        # Datetime keywords for time column detection
        self.datetime_keywords = [
            'year', 'month', 'date', 'time', 'day', 'quarter',
            'timestamp', 'period', 'when', 'temporal'
        ]
    
    def detect_intent(self, query: str) -> str:
        """
        Detect user intent from natural language query.
        
        Analyzes query keywords and returns the most likely intent:
        - "compare": Categorical group comparisons
        - "trend": Time-series trend analysis
        - "distribution": Histogram or descriptive statistics
        - "relationship": Correlation analysis
        - "predict": ML model prediction
        - "risk": Risk group identification
        - "summary": General dataset summary
        
        Args:
            query: Natural language query string
            
        Returns:
            Intent string (one of the 7 types above)
            
        Examples:
            >>> engine.detect_intent("compare male and female income")
            'compare'
            >>> engine.detect_intent("show trend of sales over time")
            'trend'
            >>> engine.detect_intent("distribution of age")
            'distribution'
        """
        query_lower = query.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Longer keywords get higher weight (more specific)
                    score += len(keyword.split())
            intent_scores[intent] = score
        
        # Return intent with highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # Default to summary if no clear intent
        return 'summary'
    
    def extract_columns(self, query: str) -> List[str]:
        """
        Extract column names mentioned in the query.
        
        Uses partial matching to map query words to actual column names.
        Supports:
        - Exact matches
        - Partial matches ("income level" → "income")
        - Datetime keywords → datetime columns
        
        Args:
            query: Natural language query string
            
        Returns:
            List of matched column names from the dataset
            
        Examples:
            >>> engine.extract_columns("compare male and female income levels")
            ['gender', 'income']
            >>> engine.extract_columns("trend of sales by year")
            ['sales', 'year']
        """
        query_lower = query.lower()
        matched_columns = []
        
        # Split query into tokens
        tokens = re.findall(r'\b\w+\b', query_lower)
        
        # Try exact column name matches first
        for col in self.columns:
            col_lower = col.lower()
            
            # Exact match
            if col_lower in query_lower:
                if col not in matched_columns:
                    matched_columns.append(col)
                continue
            
            # Partial match (column name appears as substring in query)
            col_normalized = col_lower.replace('_', ' ').replace('-', ' ')
            if col_normalized in query_lower:
                if col not in matched_columns:
                    matched_columns.append(col)
                continue
        
        # Try fuzzy matching for each token against column names
        for token in tokens:
            # Skip common words
            if token in ['the', 'and', 'or', 'of', 'by', 'in', 'on', 'at', 
                        'to', 'for', 'with', 'from', 'show', 'give', 'get']:
                continue
            
            for col_key, col_name in self.column_map.items():
                # Match if token appears in column name
                if token in col_key.split():
                    if col_name not in matched_columns:
                        matched_columns.append(col_name)
        
        # Special handling for datetime columns
        has_datetime_keyword = any(kw in query_lower for kw in self.datetime_keywords)
        
        if has_datetime_keyword:
            # Find datetime columns in dataset
            import pandas as pd
            datetime_cols = []
            
            for col in self.df.columns:
                # Check if column is datetime type
                if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    datetime_cols.append(col)
                # Try parsing as datetime
                elif self.df[col].dtype == 'object':
                    try:
                        pd.to_datetime(self.df[col], errors='raise')
                        datetime_cols.append(col)
                    except:
                        pass
            
            # Add first datetime column if found
            if datetime_cols and datetime_cols[0] not in matched_columns:
                matched_columns.append(datetime_cols[0])
        
        return matched_columns
    
    def route(self, query: str) -> Dict:
        """
        Route natural language query to appropriate analysis engine.
        
        Combines intent detection and column extraction to generate
        routing instructions for the StatFlow AI backend.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with routing information:
            {
                "intent": str (detected intent),
                "columns": List[str] (extracted column names),
                "action": str (engine.method to call),
                "params": Dict (parameters for the action),
                "confidence": float (0-1, routing confidence score)
            }
            
        Routing Map:
            - compare → analysis.crosstab
            - trend → forecasting.timeseries
            - distribution → analysis.descriptive
            - relationship → analysis.correlation
            - predict → ml.autoselect
            - risk → insight.risk_groups
            - summary → analysis.summary
            
        Examples:
            >>> engine.route("compare male and female income")
            {
                "intent": "compare",
                "columns": ["gender", "income"],
                "action": "analysis.crosstab",
                "params": {
                    "categorical_col": "gender",
                    "numeric_col": "income"
                }
            }
        """
        # Detect intent
        intent = self.detect_intent(query)
        
        # Extract columns
        columns = self.extract_columns(query)
        
        # Build routing decision
        routing = {
            "intent": intent,
            "columns": columns,
            "action": None,
            "params": {},
            "confidence": 0.0,
            "query": query
        }
        
        # Route based on intent
        if intent == "compare":
            routing["action"] = "analysis.crosstab"
            
            # Try to identify categorical and numeric columns
            if len(columns) >= 2:
                import pandas as pd
                categorical_cols = [
                    col for col in columns 
                    if self.df[col].dtype == 'object' or self.df[col].nunique() < 20
                ]
                numeric_cols = [
                    col for col in columns
                    if pd.api.types.is_numeric_dtype(self.df[col])
                ]
                
                if categorical_cols:
                    routing["params"]["categorical_col"] = categorical_cols[0]
                if numeric_cols:
                    routing["params"]["numeric_col"] = numeric_cols[0]
                
                routing["confidence"] = 0.8
            else:
                routing["confidence"] = 0.5
        
        elif intent == "trend":
            routing["action"] = "forecasting.timeseries"
            
            # Identify time and value columns
            import pandas as pd
            datetime_cols = [
                col for col in columns
                if pd.api.types.is_datetime64_any_dtype(self.df[col])
            ]
            numeric_cols = [
                col for col in columns
                if pd.api.types.is_numeric_dtype(self.df[col])
            ]
            
            if datetime_cols:
                routing["params"]["time_col"] = datetime_cols[0]
            if numeric_cols:
                routing["params"]["value_col"] = numeric_cols[0]
            
            routing["confidence"] = 0.7 if datetime_cols else 0.4
        
        elif intent == "distribution":
            routing["action"] = "analysis.descriptive"
            
            # Use first column or all numeric columns
            import pandas as pd
            if columns:
                numeric_cols = [
                    col for col in columns
                    if pd.api.types.is_numeric_dtype(self.df[col])
                ]
                if numeric_cols:
                    routing["params"]["columns"] = numeric_cols
                    routing["confidence"] = 0.8
                else:
                    routing["params"]["columns"] = columns
                    routing["confidence"] = 0.6
            else:
                # No columns specified, use all numeric
                routing["params"]["columns"] = self.df.select_dtypes(include=['number']).columns.tolist()
                routing["confidence"] = 0.5
        
        elif intent == "relationship":
            routing["action"] = "analysis.correlation"
            
            # Need at least 2 numeric columns
            import pandas as pd
            numeric_cols = [
                col for col in columns
                if pd.api.types.is_numeric_dtype(self.df[col])
            ]
            
            if len(numeric_cols) >= 2:
                routing["params"]["columns"] = numeric_cols[:2]
                routing["confidence"] = 0.8
            elif len(numeric_cols) == 1:
                # Use with all other numeric columns
                routing["params"]["columns"] = [numeric_cols[0]]
                routing["confidence"] = 0.6
            else:
                # Use all numeric columns
                routing["params"]["columns"] = self.df.select_dtypes(include=['number']).columns.tolist()[:5]
                routing["confidence"] = 0.4
        
        elif intent == "predict":
            routing["action"] = "ml.autoselect"
            
            # Try to identify target column (last mentioned or with keywords)
            target_keywords = ['target', 'outcome', 'predict', 'label', 'class', 'y']
            target_col = None
            
            # Check columns for target keywords
            for col in columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in target_keywords):
                    target_col = col
                    break
            
            # If not found, use last mentioned column
            if not target_col and columns:
                target_col = columns[-1]
            
            if target_col:
                routing["params"]["target_col"] = target_col
                routing["params"]["feature_cols"] = [c for c in columns if c != target_col]
                routing["confidence"] = 0.7
            else:
                routing["confidence"] = 0.3
        
        elif intent == "risk":
            routing["action"] = "insight.risk_groups"
            
            # Use specified columns or all columns
            if columns:
                routing["params"]["columns"] = columns
                routing["confidence"] = 0.7
            else:
                routing["params"]["columns"] = self.columns[:10]  # Limit to first 10
                routing["confidence"] = 0.5
        
        elif intent == "summary":
            routing["action"] = "analysis.summary"
            
            # Summary doesn't need specific columns
            if columns:
                routing["params"]["columns"] = columns
                routing["confidence"] = 0.8
            else:
                routing["confidence"] = 0.6
        
        return routing
    
    def validate_routing(self, routing: Dict) -> Dict:
        """
        Validate routing decision and add warnings if needed.
        
        Args:
            routing: Routing dictionary from route() method
            
        Returns:
            Enhanced routing dict with validation status and warnings
        """
        validation = {
            "valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check if columns exist
        missing_cols = [
            col for col in routing.get("columns", [])
            if col not in self.columns
        ]
        if missing_cols:
            validation["valid"] = False
            validation["warnings"].append(f"Columns not found: {missing_cols}")
        
        # Check confidence threshold
        if routing.get("confidence", 0) < 0.5:
            validation["warnings"].append(
                "Low confidence routing. Query may be ambiguous."
            )
            validation["suggestions"].append(
                "Try rephrasing with more specific column names or intent keywords."
            )
        
        # Check if enough columns for action
        action = routing.get("action", "")
        columns = routing.get("columns", [])
        
        if action == "analysis.crosstab" and len(columns) < 2:
            validation["warnings"].append(
                "Crosstab analysis requires at least 2 columns."
            )
            validation["suggestions"].append(
                "Specify both a categorical and numeric column."
            )
        
        if action == "analysis.correlation" and len(columns) < 2:
            validation["warnings"].append(
                "Correlation analysis requires at least 2 numeric columns."
            )
        
        # Add validation to routing
        routing["validation"] = validation
        
        return routing
    
    def log_query(
        self, 
        file_id: str,
        query: str, 
        intent_data: Dict, 
        narrative: str
    ) -> None:
        """
        Log natural language query to file for audit trail and PDF reporting.
        
        Creates a structured log entry and appends it to the file-specific
        NLQ log file. Handles file creation, JSON serialization, and errors
        silently to avoid breaking the main query flow.
        
        Args:
            file_id: Unique file identifier
            query: Original natural language query text
            intent_data: Routing dictionary from route() method
            narrative: Generated narrative response text
            
        Log Entry Structure:
            {
                "query": str,
                "intent": str,
                "action": str,
                "columns": List[str],
                "narrative": str,
                "confidence": float,
                "timestamp": str (ISO format)
            }
            
        Storage:
            temp_uploads/logs/<file_id>_nlq_log.json
        """
        try:
            # Ensure logs directory exists
            logs_dir = Path("temp_uploads") / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Path to NLQ log file
            log_path = logs_dir / f"{file_id}_nlq_log.json"
            
            # Load existing log or create new list
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        nlq_log = json.load(f)
                    
                    # Ensure it's a list
                    if not isinstance(nlq_log, list):
                        nlq_log = []
                except (json.JSONDecodeError, ValueError):
                    # Corrupted file, start fresh
                    nlq_log = []
            else:
                nlq_log = []
            
            # Build log entry
            log_entry = {
                "query": query,
                "intent": intent_data.get("intent", "unknown"),
                "action": intent_data.get("action", "N/A"),
                "columns": intent_data.get("columns", []),
                "narrative": narrative,
                "confidence": intent_data.get("confidence", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Append new entry
            nlq_log.append(log_entry)
            
            # Save updated log with JSON-safe serialization
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(
                    nlq_log, 
                    f, 
                    indent=2, 
                    ensure_ascii=False,
                    default=str  # Handle non-serializable types
                )
        
        except Exception as e:
            # Silent failure - log to console but don't break system
            print(f"Warning: NLQ logging failed for file_id={file_id}: {e}")
            # Could optionally log to system audit log here
