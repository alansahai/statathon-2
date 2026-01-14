"""
GenAI Engine – Clean, Fixed, Updated Version
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenAIEngine")


class GenAIError(Exception):
    """Custom exception for GenAI-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_body: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
    
    def __str__(self):
        parts = [self.message]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        return " | ".join(parts)


class GenAIEngine:

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_GENAI_API_KEY")

        if not self.api_key:
            raise GenAIError("GOOGLE_GENAI_API_KEY missing")

        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.0-flash-exp"
        self.event_log = []

    # ---------------------------------------------------------------------
    # EVENT LOGGING
    # ---------------------------------------------------------------------
    def log_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an event with timestamp and structured details."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        log_entry = {
            "timestamp": timestamp,
            "event": event_type,
            "message": message
        }
        if details:
            log_entry["details"] = details
        
        self.event_log.append(log_entry)
        logger.info(f"[{event_type}] {message}")

    # ---------------------------------------------------------------------
    # FIXED BUILD PROMPT (with self)
    # ---------------------------------------------------------------------
    def build_prompt(self, summary_dict: Dict[str, Any]) -> str:
        """
        Optimized prompt for long-form narrative generation.
        Forces 3 paragraphs with strict word count requirements.
        """

        dataset = summary_dict.get("dataset_overview", {})
        key_stats = summary_dict.get("key_stats", {})
        numeric = key_stats.get("numeric_stats", {})
        freq = key_stats.get("frequencies", {})

        data_lines = []
        data_lines.append(f"Dataset size: {dataset.get('rows')} rows, {dataset.get('columns')} columns.")

        data_lines.append("\nNUMERIC VARIABLES:")
        for col, stats in numeric.items():
            data_lines.append(
                f"- {col}: mean={stats.get('mean')}, median={stats.get('median')}, std={stats.get('std')}"
            )

        data_lines.append("\nCATEGORICAL VARIABLES:")
        for col, block in freq.items():
            data_lines.append(f"- {col}: proportions={block.get('proportions')}")

        data_block = "\n".join(data_lines)

        prompt = f"""You are a statistical analyst. Your task is to write a long-form narrative analysis based ONLY on the dataset summary provided below. Follow ALL rules strictly.

RULES:
1. Do NOT invent numbers or percentages.
2. Only discuss values that appear in the dataset summary.
3. Do NOT calculate new correlations or relationships.
4. Do NOT use bullet points or lists.
5. Write exactly 3 paragraphs.
6. Each paragraph must contain at least 120–150 words.
7. Total length must be 350–500 words.
8. Use formal analytical tone.
9. Include interpretation, not speculation.
10. Mention sample size and limitations if applicable.

DATA SUMMARY:
{data_block}

Now generate the narrative.
Do NOT include headings. Write continuous paragraphs only.

---

After writing the full 3-paragraph narrative, output the same content again in a structured JSON-like format with the following keys:

{{
  "executive_summary": "<3–4 sentences summarizing the dataset>",
  "numeric_insights": "<explanation of numeric variable trends>",
  "categorical_insights": "<explanation of distributions and patterns>",
  "sample_limitations": "<explanation of sample size or constraints>",
  "strategic_recommendations": "<practical next steps based on the insights>"
}}

Rules for structured output:
- Do NOT invent new numbers.
- Do NOT add percentages not present in the summary.
- Do NOT hallucinate correlations.
- KEEP the content consistent with the narrative above.
- Do NOT include markdown.
- Do NOT wrap the JSON in backticks."""

        return prompt


    # ---------------------------------------------------------------------
    # FIXED LLM CALL (correct Gemini v1 syntax)
    # ---------------------------------------------------------------------
    def call_llm(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            logger.info("RAW RESPONSE:")
            logger.info(str(response))

            if not response or not response.text:
                return "LLM returned empty response"

            full_text = response.text.strip()

            if len(full_text) < 100:
                return "LLM returned too little text"

            return full_text

        except Exception as e:
            logger.error(f"LLM ERROR: {e}")
            return f"GENAI ERROR: {str(e)}"

    # ---------------------------------------------------------------------
    # JSON EXTRACTION
    # ---------------------------------------------------------------------
    def extract_structured_json(self, text: str) -> Optional[dict]:
        """
        Detects and extracts the JSON-like structure appended after the narrative.
        Returns a dict or None.
        """
        try:
            # isolate anything starting with { and ending with }
            json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
            if not json_match:
                return None

            raw_json = json_match.group(0)

            # fix common LLM mistakes like trailing commas
            raw_json = re.sub(r',\s*}', '}', raw_json)
            raw_json = re.sub(r',\s*\]', ']', raw_json)

            return json.loads(raw_json)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # RESPONSE NORMALIZATION
    # ---------------------------------------------------------------------
    def normalize_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Normalize raw narrative text into structured format.
        Extracts JSON if present, otherwise returns basic structure.
        """
        # Try extracting JSON structure first
        structured = self.extract_structured_json(raw_text)
        
        if structured:
            return structured
        
        # Fallback: create basic structure from raw text
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
        
        return {
            "executive_summary": paragraphs[0] if len(paragraphs) > 0 else raw_text[:300],
            "numeric_insights": paragraphs[1] if len(paragraphs) > 1 else "Not available",
            "categorical_insights": paragraphs[2] if len(paragraphs) > 2 else "Not available",
            "sample_limitations": "Not available",
            "strategic_recommendations": "Not available"
        }

    # ---------------------------------------------------------------------
    # PUBLIC API: Generate narrative
    # ---------------------------------------------------------------------
    def generate_narrative(self, summary_dict):
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = self.build_prompt(summary_dict)

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-pro",
                contents=[{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }]
            )

            # Extract the text from response
            candidate = response.candidates[0]
            parts = candidate.content.parts
            text = " ".join([p.text for p in parts if hasattr(p, "text")]).strip()

            # Validate length
            if len(text) < 200:
                return (
                    "Narrative generation completed, but response was too short "
                    "(<200 chars). Try a larger model or simplified prompt."
                )

            return text

        except Exception as e:
            return f"GENAI ERROR: {str(e)}"

    def generate_narrative_with_prompt(self, custom_prompt: str) -> str:
        """
        Generate narrative using a custom prompt (for regeneration).
        """
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-pro",
                contents=[{
                    "role": "user",
                    "parts": [{"text": custom_prompt}]
                }]
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts
            text = " ".join([p.text for p in parts if hasattr(p, "text")]).strip()

            return text if len(text) >= 200 else "Regeneration produced insufficient text"

        except Exception as e:
            return f"GENAI ERROR: {str(e)}"

    # ---------------------------------------------------------------------
    # HALLUCINATION DETECTION
    # ---------------------------------------------------------------------
    def _detect_hallucinations(self, narrative_json: Dict[str, Any], summary_dict: Dict[str, Any], full_text: str) -> list:
        """
        Upgraded hallucination detector with expanded number formats,
        semantic checks, and contextual reporting.
        """
        # SMART NUMBER MATCHING WINDOW
        INT_TOLERANCE = 1.0          # allow ±1 for integer fields
        FLOAT_TOLERANCE = 0.05       # allow ±0.05 for floats
        PERCENT_TOLERANCE = 0.5      # allow ±0.5% for proportions
        RANGE_TOLERANCE = 0.1        # allow 10% variation for ranges
        IGNORE_SMALL_VALUES = True   # ignore integers < 10 unless explicitly categorical
        
        issues = []
        
        # Extract all valid numbers from summary_dict
        valid_numbers = set()
        valid_percentages = set()  # Track percentages separately
        
        def extract_all_numbers(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    # If we're in a 'proportions' dict, convert values to percentages
                    if 'proportions' in path.lower():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            valid_percentages.add(float(v) * 100)  # Convert 0.52 to 52%
                            valid_numbers.add(float(v))  # Also keep decimal form
                    extract_all_numbers(v, new_path)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_all_numbers(item, path)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                valid_numbers.add(float(obj))
        
        extract_all_numbers(summary_dict)
        
        # Helper functions for smart matching
        def is_close(num, valid_set):
            """Check if number is close enough to any valid number."""
            for v in valid_set:
                # Fixed tolerance for integers
                if abs(v - num) <= INT_TOLERANCE:
                    return True
                # Relative tolerance for floats
                if abs(v - num) <= FLOAT_TOLERANCE * max(1, abs(v)):
                    return True
            return False
        
        def is_valid_percentage(num, valid_percentages):
            """Check if percentage is within tolerance."""
            for p in valid_percentages:
                if abs(p - num) <= PERCENT_TOLERANCE:
                    return True
            return False
        
        # Extract numbers from narrative with multiple formats
        def extract_narrative_numbers(text):
            found = []
            percentage_positions = set()  # Track positions already matched as percentages
            comma_positions = set()  # Track positions already matched as comma numbers
            range_positions = set()  # Track positions already matched as ranges
            
            # Comma-formatted numbers: 65,400
            for match in re.finditer(r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b', text):
                num_str = match.group(1).replace(',', '')
                pos = match.start()
                comma_positions.add(pos)
                found.append(('number', float(num_str), pos, match.group(0)))
            
            # Percentages: 10%, 40.5%
            for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%', text):
                num = float(match.group(1))
                pos = match.start()
                percentage_positions.add(pos)
                found.append(('percentage', num, pos, match.group(0)))
            
            # Ranges: 20-30 or 20–30
            for match in re.finditer(r'\b(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\b', text):
                num1 = float(match.group(1))
                num2 = float(match.group(2))
                pos = match.start()
                range_positions.add(pos)
                found.append(('range', (num1, num2), pos, match.group(0)))
            
            # Regular numbers (skip if already matched as percentage, comma-number, or range)
            for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
                pos = match.start()
                # Skip if this position is already part of a percentage, comma number, or range
                if pos in percentage_positions or pos in comma_positions or pos in range_positions:
                    continue
                if not re.match(r'\d{1,3}(?:,\d{3})+', match.group(0)):
                    num = float(match.group(1))
                    found.append(('number', num, pos, match.group(0)))
            
            return found
        
        def get_context_snippet(text, pos, length=40):
            start = max(0, pos - length)
            end = min(len(text), pos + length)
            return text[start:end]
        
        def check_number_validity(num_type, value, pos, original_str):
            if num_type == 'percentage':
                # Ignore small percentages < 10 if enabled
                if IGNORE_SMALL_VALUES and value < 10:
                    return None
                
                # Check against valid percentages first
                if is_valid_percentage(value, valid_percentages):
                    return None
                
                # Check if percentage value exists in valid numbers (already as %)
                if is_close(value, valid_numbers):
                    return None
                
                # Check if decimal equivalent exists (40% = 0.4)
                decimal_val = value / 100.0
                if is_close(decimal_val, valid_numbers):
                    return None
                
                return {
                    "section": "unknown",
                    "category": "percentage",
                    "value": original_str,
                    "details": f"Percentage {value}% not found in summary (tolerance: ±{PERCENT_TOLERANCE}pp)",
                    "context": get_context_snippet(full_text, pos)
                }
            
            elif num_type == 'range':
                num1, num2 = value
                # Both ends must be close to known values
                found1 = is_close(num1, valid_numbers)
                found2 = is_close(num2, valid_numbers)
                if not (found1 and found2):
                    return {
                        "section": "unknown",
                        "category": "number",
                        "value": original_str,
                        "details": f"Range {num1}-{num2} contains invalid endpoints",
                        "context": get_context_snippet(full_text, pos)
                    }
                return None
            
            else:  # regular number
                # Ignore small integers < 10 if enabled
                if IGNORE_SMALL_VALUES and value < 10 and value == int(value):
                    return None
                
                # Use smart matching with improved tolerance
                if is_close(value, valid_numbers):
                    return None
                
                return {
                    "section": "unknown",
                    "category": "number",
                    "value": original_str,
                    "details": f"Number {value} not found in summary (tolerance: ±{INT_TOLERANCE if value == int(value) else FLOAT_TOLERANCE})",
                    "context": get_context_snippet(full_text, pos)
                }
        
        # Check all extracted numbers
        narrative_numbers = extract_narrative_numbers(full_text)
        for num_type, value, pos, original_str in narrative_numbers:
            issue = check_number_validity(num_type, value, pos, original_str)
            if issue:
                issues.append(issue)
        
        # Cross-validate numeric_stats fields
        key_stats = summary_dict.get("key_stats", {})
        numeric_stats = key_stats.get("numeric_stats", {})
        
        for section_key, section_text in narrative_json.items():
            if not isinstance(section_text, str):
                continue
            
            # Check for mean/median/std references
            stat_patterns = [
                (r'mean[s]?\s+(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)', 'mean'),
                (r'median[s]?\s+(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)', 'median'),
                (r'standard deviation[s]?\s+(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)', 'std'),
                (r'average[s]?\s+(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)', 'mean')
            ]
            
            for pattern, stat_name in stat_patterns:
                for match in re.finditer(pattern, section_text, re.IGNORECASE):
                    mentioned_value = float(match.group(1))
                    found_in_stats = False
                    
                    for var_name, stats in numeric_stats.items():
                        if stat_name in stats:
                            valid_val = stats[stat_name]
                            if abs(mentioned_value - valid_val) <= 0.01:
                                found_in_stats = True
                                break
                    
                    if not found_in_stats:
                        issues.append({
                            "section": section_key,
                            "category": "number",
                            "value": f"{stat_name}={mentioned_value}",
                            "details": f"Mentioned {stat_name} value not found in numeric_stats",
                            "context": match.group(0)
                        })
        
        # Cross-validate frequencies
        frequencies = key_stats.get("frequencies", {})
        for section_key, section_text in narrative_json.items():
            if not isinstance(section_text, str):
                continue
            
            # Check for category mentions
            for var_name, freq_data in frequencies.items():
                if isinstance(freq_data, dict) and 'proportions' in freq_data:
                    proportions = freq_data['proportions']
                    if isinstance(proportions, dict):
                        for category in proportions.keys():
                            # Ensure category is actually mentioned if numbers about it appear
                            category_pattern = re.escape(str(category))
                            if re.search(category_pattern, section_text, re.IGNORECASE):
                                # Category is mentioned, which is good
                                pass
        
        # Semantic hallucination detection
        causal_patterns = [
            r'\b(?:due to|because of|caused by|resulted? in|leads? to|attributed to)\b.*\b(?:increase|decrease|change|higher|lower)\b',
            r'\b(?:therefore|thus|hence|consequently)\b.*\b(?:increase|decrease|change|affect|higher|lower)',
        ]
        
        # Correlation patterns (more specific - requires explicit correlation/correlation claims)
        correlation_patterns = [
            r'\b(?:correlation|correlated|correlates|correlation coefficient)\b',
            r'\b(?:strong|weak|significant|positive|negative)\s+(?:correlation|association|relationship)\b.*\b(?:between|with)\b',
        ]
        
        for section_key, section_text in narrative_json.items():
            if not isinstance(section_text, str):
                continue
            
            # Skip semantic checks for limitations and recommendations sections
            if 'limitation' in section_key.lower() or 'recommendation' in section_key.lower():
                continue
            
            # Check causal claims
            for pattern in causal_patterns:
                for match in re.finditer(pattern, section_text, re.IGNORECASE):
                    pos = match.start()
                    issues.append({
                        "section": section_key,
                        "category": "causal",
                        "value": match.group(0),
                        "details": "Causal claim detected - not supported by descriptive statistics",
                        "context": get_context_snippet(section_text, pos)
                    })
            
            # Check correlation claims
            for pattern in correlation_patterns:
                for match in re.finditer(pattern, section_text, re.IGNORECASE):
                    pos = match.start()
                    issues.append({
                        "section": section_key,
                        "category": "semantic",
                        "value": match.group(0),
                        "details": "Correlation claim detected - not calculated in summary",
                        "context": get_context_snippet(section_text, pos)
                    })
        
        # Detect trend inference (more specific patterns)
        trend_patterns = [
            r'\b(?:trending|increasing trend|decreasing trend|upward trend|downward trend)\b',
            r'\b(?:will|future|predict|forecast|projection)\b',
        ]
        
        for section_key, section_text in narrative_json.items():
            if not isinstance(section_text, str):
                continue
            
            # Skip semantic checks for limitations and recommendations sections
            if 'limitation' in section_key.lower() or 'recommendation' in section_key.lower():
                continue
            
            for pattern in trend_patterns:
                for match in re.finditer(pattern, section_text, re.IGNORECASE):
                    pos = match.start()
                    issues.append({
                        "section": section_key,
                        "category": "semantic",
                        "value": match.group(0),
                        "details": "Temporal/trend inference not supported by cross-sectional data",
                        "context": get_context_snippet(section_text, pos)
                    })
        
        # Detect invented outliers
        outlier_patterns = [
            r'\b(?:outlier|anomal(?:y|ies)|extreme value|unusual)\b'
        ]
        
        dataset_overview = summary_dict.get("dataset_overview", {})
        has_outlier_info = "outliers" in str(dataset_overview).lower()
        
        if not has_outlier_info:
            for section_key, section_text in narrative_json.items():
                if not isinstance(section_text, str):
                    continue
                
                for pattern in outlier_patterns:
                    for match in re.finditer(pattern, section_text, re.IGNORECASE):
                        pos = match.start()
                        issues.append({
                            "section": section_key,
                            "category": "semantic",
                            "value": match.group(0),
                            "details": "Outlier mention without outlier detection in summary",
                            "context": get_context_snippet(section_text, pos)
                        })
        
        return issues

    # ---------------------------------------------------------------------
    # AUTO-REGENERATION (Upgrade A)
    # ---------------------------------------------------------------------
    def auto_regenerate(self, summary_dict: Dict[str, Any], validation_report: Dict[str, Any], attempt: int) -> str:
        """
        Automatically regenerates narrative using stricter prompts when validation fails.
        attempt = 1 or 2
        """
        if attempt == 1:
            # First rewrite - strict filtering
            prompt = (
                "Rewrite the narrative with the following strict conditions:\n"
                "- No new numbers\n"
                "- No causal claims\n"
                "- No temporal claims\n"
                "- Replace uncertainties with descriptive statements\n"
                "- Expand each section to 120–160 words\n"
            )
        else:
            # Final fallback prompt - minimal safe summary
            prompt = (
                "Generate a simple descriptive summary of the dataset:\n"
                "1. Overview\n"
                "2. Numeric patterns\n"
                "3. Category patterns\n"
                "Avoid conclusions, predictions, correlations, trends, or causes."
            )

        prompt += "\n\nDATA:\n" + json.dumps(summary_dict, indent=2)

        regenerated = self.generate_narrative_with_prompt(prompt)
        return regenerated

    # ---------------------------------------------------------------------
    # NARRATIVE VALIDATION
    # ---------------------------------------------------------------------
    def validate_narrative(self, narrative_json: Dict[str, Any], summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates narrative for hallucinations, format issues, and completeness.
        Returns structured validation report.
        """
        issues = []
        
        # Concatenate all text for full analysis
        all_text = ""
        for key, value in narrative_json.items():
            if isinstance(value, str):
                all_text += value + " "
            elif isinstance(value, list):
                all_text += " ".join(str(item) for item in value) + " "
        
        # Run hallucination detection
        hallucinations = self._detect_hallucinations(narrative_json, summary_dict, all_text)
        
        for hallucination in hallucinations:
            issues.append({
                "section": hallucination["section"],
                "problem": "hallucination",
                "category": hallucination["category"],
                "value": hallucination["value"],
                "details": hallucination["details"],
                "context": hallucination.get("context", "")
            })
        
        # Check for completeness
        required_fields = ["executive_summary", "numeric_insights", "categorical_insights"]
        for field in required_fields:
            if field not in narrative_json or not narrative_json[field] or narrative_json[field] == "Not available":
                issues.append({
                    "section": field,
                    "problem": "missing_content",
                    "details": f"Required field '{field}' is missing or empty"
                })
        
        # Check for format violations (markdown, emojis)
        markdown_pattern = r'[#*_`\[\]]|```'
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        
        for section_key, section_text in narrative_json.items():
            if not isinstance(section_text, str):
                continue
            
            if re.search(markdown_pattern, section_text):
                issues.append({
                    "section": section_key,
                    "problem": "format_violation",
                    "details": "Contains markdown formatting"
                })
            
            if re.search(emoji_pattern, section_text):
                issues.append({
                    "section": section_key,
                    "problem": "format_violation",
                    "details": "Contains emojis"
                })
        
        # Word count check
        word_count = len(all_text.split())
        if word_count < 300:
            issues.append({
                "section": "overall",
                "problem": "length_violation",
                "details": f"Total word count {word_count} is below minimum 300 words"
            })
        
        # Determine overall status
        has_hallucinations = any(issue["problem"] == "hallucination" for issue in issues)
        has_missing_content = any(issue["problem"] == "missing_content" for issue in issues)
        has_format_issues = any(issue["problem"] == "format_violation" for issue in issues)
        
        if has_hallucinations:
            overall_status = "unsafe"
        elif has_missing_content or has_format_issues:
            overall_status = "needs_revision"
        elif len(issues) > 0:
            overall_status = "needs_revision"
        else:
            overall_status = "ok"
        
        return {
            "is_valid": len(issues) == 0,
            "overall_status": overall_status,
            "issues": issues,
            "word_count": word_count,
            "hallucination_count": sum(1 for i in issues if i["problem"] == "hallucination"),
            "sections_validated": len([k for k, v in narrative_json.items() if isinstance(v, (str, list))])
        }

    # ---------------------------------------------------------------------
    # SECTION PARTITIONING (Upgrade B)
    # ---------------------------------------------------------------------
    def partition_narrative(self, raw_text: str) -> Dict[str, Optional[str]]:
        """
        Splits narrative into structured sections even if the LLM didn't properly format them.
        """
        paragraphs = [p.strip() for p in raw_text.split("\n") if len(p.strip()) > 40]

        return {
            "executive_summary": paragraphs[0] if len(paragraphs) > 0 else None,
            "numeric_insights": paragraphs[1] if len(paragraphs) > 1 else None,
            "categorical_insights": paragraphs[2] if len(paragraphs) > 2 else None,
            "risks": paragraphs[3] if len(paragraphs) > 3 else None,
            "recommendations": paragraphs[4] if len(paragraphs) > 4 else None
        }

    # ---------------------------------------------------------------------
    # CONFIDENCE SCORING (Upgrade C)
    # ---------------------------------------------------------------------
    def compute_confidence(self, validation_report: Dict[str, Any]) -> float:
        """
        Computes confidence score (0.0 to 1.0) based on validation issues.
        """
        issues = len(validation_report["issues"])
        hallucinations = sum(1 for x in validation_report["issues"] if x["problem"] == "hallucination")

        score = 1.0 - ((issues * 0.05) + (hallucinations * 0.1))
        score = max(0.0, min(1.0, score))  # clamp to 0–1
        return round(score, 3)

    # ---------------------------------------------------------------------
    # NARRATIVE COMPRESSION (Upgrade D)
    # ---------------------------------------------------------------------
    def compress_narrative(self, text: str, target_words: int = 250) -> str:
        """
        Compresses narrative to target word count using intelligent summarization.
        """
        words = text.split()
        if len(words) <= target_words:
            return text

        # Basic summarization heuristic: take first N% of each paragraph
        paragraphs = text.split("\n")
        compressed = []

        slice_len = target_words // max(1, len(paragraphs))

        for p in paragraphs:
            p_words = p.split()
            compressed.append(" ".join(p_words[:slice_len]))

        return "\n".join(compressed)

    # ---------------------------------------------------------------------
    # TWO-PASS VALIDATION GATE
    # ---------------------------------------------------------------------
    def generate_validated_narrative(self, summary_dict):
        """
        Two-Pass Validation Gate
        1) Generate raw narrative
        2) Normalize output and validate for hallucinations, formatting, length
        Returns structured, safe output with status flags.
        """

        self.log_event("GATE_PASS1", "Starting narrative generation")

        # PASS 1 — raw narrative generation
        raw_narrative = self.generate_narrative(summary_dict)

        self.log_event("GATE_VALIDATE", "Running validation layer")

        # PASS 2 — structured + safe validation
        structured = self.normalize_response(raw_narrative)
        validation = self.validate_narrative(structured, summary_dict)

        hallucination_count = sum(
            1 for issue in validation["issues"] if issue["problem"] == "hallucination"
        )

        # Determine status
        if hallucination_count > 0:
            status = "unsafe"
        elif validation["overall_status"] == "needs_revision":
            status = "needs_revision"
        else:
            status = "ok"

        # Log final gate decision
        self.log_event(
            "GATE_DECISION",
            f"Narrative classified as {status}",
            {
                "hallucinations": hallucination_count,
                "word_count": validation["word_count"],
                "issues": len(validation["issues"])
            }
        )

        return {
            "status": status,
            "raw_narrative": raw_narrative,
            "structured_narrative": structured,
            "validation_report": validation
        }


    # ---------------------------------------------------------------------
    # STRUCTURED JSON WRAPPER WITH AUTO-REGENERATION LOOP
    # ---------------------------------------------------------------------
    def generate_structured_narrative(self, summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates narrative with automatic regeneration loop.
        Tries up to 3 passes to produce valid, high-quality output.
        """
        LOG = self.log_event

        # PASS 1 — generate normally
        LOG("GATE_PASS1", "Starting narrative generation")
        raw = self.generate_narrative(summary_dict)
        normalized = self.normalize_response(raw)

        # PASS 1 — validate
        LOG("GATE_VALIDATE", "Running validation layer")
        report = self.validate_narrative(normalized, summary_dict)

        # Apply compression if small dataset
        row_count = summary_dict.get("dataset_overview", {}).get("rows", 1000)
        
        # If OK, return immediately with partitioning and confidence
        if report["overall_status"] == "ok":
            final_narrative = raw
            if row_count < 30:
                final_narrative = self.compress_narrative(raw, target_words=180)
            
            # Apply partitioning
            partitioned = self.partition_narrative(final_narrative)
            
            return {
                "narrative": final_narrative,
                "normalized": normalized,
                "partitioned": partitioned,
                "validation": report,
                "confidence": self.compute_confidence(report),
                "status": "ok",
                "timestamp": datetime.now().isoformat()
            }

        # PASS 2 — auto-regenerate (strict filtering)
        LOG("GATE_REWRITE", "Attempting rewrite pass 1 (strict filtering)")
        rewritten_1 = self.auto_regenerate(summary_dict, report, attempt=1)
        normalized_2 = self.normalize_response(rewritten_1)
        report_2 = self.validate_narrative(normalized_2, summary_dict)

        if report_2["overall_status"] == "ok":
            final_narrative = rewritten_1
            if row_count < 30:
                final_narrative = self.compress_narrative(rewritten_1, target_words=180)
            
            # Apply partitioning
            partitioned = self.partition_narrative(final_narrative)
            
            return {
                "narrative": final_narrative,
                "normalized": normalized_2,
                "partitioned": partitioned,
                "validation": report_2,
                "confidence": self.compute_confidence(report_2),
                "status": "rewritten_ok",
                "timestamp": datetime.now().isoformat()
            }

        # PASS 3 — fallback rewrite (minimal safe summary)
        LOG("GATE_REWRITE_FINAL", "Attempting rewrite pass 2 (minimal safe summary)")
        rewritten_2 = self.auto_regenerate(summary_dict, report, attempt=2)
        normalized_3 = self.normalize_response(rewritten_2)
        report_3 = self.validate_narrative(normalized_3, summary_dict)
        
        final_narrative = rewritten_2
        if row_count < 30:
            final_narrative = self.compress_narrative(rewritten_2, target_words=180)
        
        # Apply partitioning
        partitioned = self.partition_narrative(final_narrative)

        return {
            "narrative": final_narrative,
            "normalized": normalized_3,
            "partitioned": partitioned,
            "validation": report_3,
            "confidence": self.compute_confidence(report_3),
            "status": "fallback_summary",
            "timestamp": datetime.now().isoformat()
        }
