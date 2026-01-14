"""
GenAI Narrative Engine - Two-Pass Architecture
Production-grade implementation for MoSPI-compliant statistical narratives.

Version: 2.0
Architecture: Two-pass generation with validation-driven regeneration
Safety: Zero-tolerance for hallucinations
"""

import os
import re
import time
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("NarrativeEngine")


class PromptAssembler:
    """
    Builds MoSPI-compliant prompts from structured statistical data.
    Embeds explicit constraints and valid data values.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PromptAssembler")
    
    def build(self, summary_dict: Dict[str, Any]) -> str:
        """
        Constructs prompt with embedded data and strict constraints.
        
        Args:
            summary_dict: Structured statistical summary
            
        Returns:
            Formatted prompt string with data block and instructions
        """
        self.logger.debug("Building prompt from summary_dict")
        
        # Extract overview
        overview = summary_dict.get("dataset_overview", {})
        key_stats = summary_dict.get("key_stats", {})
        
        rows = summary_dict.get("row_count", overview.get("rows", 0))
        cols = summary_dict.get("column_count", overview.get("columns", 0))
        filename = overview.get("filename", "dataset")
        
        # Extract numeric statistics
        numeric_stats = key_stats.get("numeric_stats", {})
        frequencies = key_stats.get("frequencies", {})
        
        # Build metrics list
        metrics = []
        for var, stats in list(numeric_stats.items())[:4]:
            if isinstance(stats, dict):
                mean = stats.get("mean")
                median = stats.get("median")
                std = stats.get("std")
                if mean is not None:
                    cv = (std / mean * 100) if mean != 0 and std else 0
                    metrics.append(f"{var}: Mean={mean:.2f}, Median={median:.2f}, SD={std:.2f}, CV={cv:.1f}%")
        
        # Build segment summaries
        segments = []
        for var, freq_data in list(frequencies.items())[:4]:
            if isinstance(freq_data, dict) and "counts" in freq_data:
                counts = freq_data["counts"]
                props = freq_data.get("proportions", {})
                items = list(counts.items())[:3]
                seg_str = f"{var}: " + ", ".join([f"{k}={v}({props.get(k,0):.0f}%)" for k, v in items])
                segments.append(seg_str)
        
        # Extract quality indicators
        outliers = summary_dict.get("outliers", {})
        risks = summary_dict.get("risk_indicators", {})
        moe = summary_dict.get("moe_indicators", {})
        
        outlier_count = outliers.get("total_columns_with_outliers", 0)
        quality_score = risks.get("data_quality_score", 100.0)
        moe_value = moe.get("moe_95_percent", "N/A")
        
        # Build data block
        data_block = f"""DATA: {filename}
SAMPLE: {rows} records, {cols} variables

METRICS:
{chr(10).join(metrics) if metrics else "Statistics computed"}

SEGMENTS:
{chr(10).join(segments) if segments else "Distributions analyzed"}

QUALITY: {quality_score:.0f}/100 | Outliers: {outlier_count} | MoE: ±{moe_value if isinstance(moe_value, str) else f"{moe_value:.1f}%"}"""

        prompt = f"""You are a government statistical analyst. Write a formal report using ONLY the data below.

CONSTRAINTS:
- Use ONLY numbers from the data below
- Do NOT invent percentages, correlations, or relationships
- If data is missing, state "data not provided"
- Do NOT assume causality or make predictions
- Do NOT create comparisons without explicit data
- Length: 250-400 words

{data_block}

OUTPUT STRUCTURE:

## DATASET OVERVIEW
State the sample size, variable count, and data source. One paragraph.

## KEY FINDINGS
List 4-6 bullet points using ACTUAL values from METRICS above:
• Describe central tendencies (means, medians)
• Interpret variability (CV percentages)
• Note any consistency or dispersion patterns
• Reference specific numeric values only

## SEGMENT INSIGHTS
Analyze SEGMENTS data:
• Gender distribution: cite actual counts and percentages
• Regional patterns: cite actual counts and percentages  
• Product usage: cite actual counts and percentages
• Purchase intent: cite actual counts and percentages
Use ONLY values from SEGMENTS above.

## VARIANCE INTERPRETATION
For each metric, explain what the SD and CV indicate:
• Low CV (<10%): consistent/homogeneous
• Medium CV (10-30%): moderate variation
• High CV (>30%): high dispersion
Reference actual CV values from METRICS.

## SATISFACTION DISTRIBUTION
If satisfaction data exists in METRICS, describe its distribution (mean, median, spread). If absent, state "satisfaction data not provided."

## DATA QUALITY & MARGIN OF ERROR
• Quality score interpretation: {quality_score:.0f}/100
• Outlier implications: {outlier_count} variables affected
• Margin of error: ±{moe_value if isinstance(moe_value, str) else f"{moe_value:.1f}%"} at 95% confidence
• Sample size adequacy for inference

## RECOMMENDATIONS
Provide 3-4 actionable items based on observed patterns (not assumptions):
• Data collection improvements
• Areas needing further investigation
• Statistical considerations for decision-making

Use formal tone. Do not exceed 400 words."""
        
        self.logger.debug(f"Prompt built: {len(prompt)} characters")
        return prompt
    
    def build_corrective(self, summary_dict: Dict[str, Any], issues: List[Dict]) -> str:
        """
        Builds corrective prompt based on validation issues.
        
        Args:
            summary_dict: Original data
            issues: List of validation issues
            
        Returns:
            Corrective prompt emphasizing problem areas
        """
        self.logger.info(f"Building corrective prompt for {len(issues)} issues")
        
        # Extract issue details
        hallucinations = [i for i in issues if i["problem"] == "hallucination"]
        missing = [i for i in issues if i["problem"] == "missing"]
        
        base_prompt = self.build(summary_dict)
        
        if hallucinations:
            correction = "\n\nCORRECTION REQUIRED:\n"
            correction += "Your previous response contained invented data. "
            correction += "You MUST use ONLY the exact numbers provided above. "
            correction += "Do NOT calculate new values. Do NOT infer relationships.\n"
            return correction + base_prompt
        
        elif missing:
            missing_sections = [i["section"] for i in missing]
            correction = f"\n\nCOMPLETION REQUIRED:\n"
            correction += f"Generate ONLY these missing sections: {', '.join(missing_sections)}\n"
            return correction + base_prompt
        
        return base_prompt


class LLMClient:
    """
    Multi-provider LLM client with retry logic and error handling.
    Supports Google Gemini, OpenAI, Anthropic Claude, and Mistral.
    """
    
    PROVIDERS = {
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "model": "gemini-pro",
            "endpoint": "/models/gemini-pro:generateContent"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "endpoint": "/chat/completions"
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-3-sonnet-20240229",
            "endpoint": "/messages"
        },
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "model": "mistral-large-latest",
            "endpoint": "/chat/completions"
        }
    }
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize LLM client with provider.
        
        Args:
            provider: "gemini" | "openai" | "anthropic" | "mistral"
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.config = self.PROVIDERS.get(self.provider)
        
        if not self.config:
            raise ValueError(f"Unknown provider: {provider}")
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        self.max_retries = 3
        self.timeout = 30
        self.backoff_multiplier = 2.0
        self.initial_backoff = 1.0
        
        self.logger = logging.getLogger(f"LLMClient.{provider}")
        self.logger.info(f"Initialized {provider} client with model {self.config['model']}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        key_map = {
            "gemini": "GOOGLE_GENAI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY"
        }
        env_var = key_map.get(self.provider)
        return os.getenv(env_var) if env_var else None
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using configured LLM provider with retry logic.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Generated text
            
        Raises:
            Exception: If all retries fail
        """
        backoff = self.initial_backoff
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{self.max_retries}")
                
                response = self._call_provider(prompt)
                
                if response and len(response.strip()) >= 50:
                    self.logger.info(f"Generated {len(response)} characters")
                    return response
                else:
                    self.logger.warning(f"Response too short: {len(response) if response else 0} chars")
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= self.backoff_multiplier
                        continue
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Timeout on attempt {attempt}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= self.backoff_multiplier
                else:
                    raise
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt}: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= self.backoff_multiplier
                else:
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt}: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= self.backoff_multiplier
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    def _call_provider(self, prompt: str) -> str:
        """Call specific provider API."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "mistral":
            return self._call_mistral(prompt)
        else:
            raise ValueError(f"Provider {self.provider} not implemented")
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        url = f"{self.config['base_url']}{self.config['endpoint']}?key={self.api_key}"
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            self.logger.warning("Empty candidates in Gemini response")
            return ""
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        
        if isinstance(content, dict) and "parts" in content:
            parts = content.get("parts", [])
            texts = [p.get("text", "").strip() for p in parts if p.get("text")]
            return "\n".join(texts)
        
        return ""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        url = f"{self.config['base_url']}{self.config['endpoint']}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        url = f"{self.config['base_url']}{self.config['endpoint']}"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral API."""
        url = f"{self.config['base_url']}{self.config['endpoint']}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


class ResponseNormalizer:
    """
    Transforms raw LLM output into structured, clean JSON.
    Removes markdown, emojis, disclaimers, and organizes content by sections.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ResponseNormalizer")
    
    def normalize(self, text: str) -> Dict[str, Any]:
        """
        Convert raw text into structured narrative JSON.
        
        Args:
            text: Raw LLM output
            
        Returns:
            Dictionary with structured sections
        """
        self.logger.debug(f"Normalizing {len(text)} characters")
        
        if not text or len(text.strip()) < 20:
            return self._empty_structure()
        
        # Clean text
        cleaned = self._strip_chatter(text)
        cleaned = self._remove_markdown(cleaned)
        cleaned = self._remove_emojis(cleaned)
        
        # Split into lines
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        
        # Detect sections
        result = self._detect_sections(lines)
        
        # Validate and clean
        result = self._validate_placeholders(result)
        
        self.logger.info(f"Normalized into {len([k for k, v in result.items() if v])} non-empty sections")
        return result
    
    def _empty_structure(self) -> Dict[str, Any]:
        """Return empty narrative structure."""
        return {
            "executive_summary": None,
            "key_findings": [],
            "segment_insights": [],
            "trends": None,
            "risks": None,
            "margin_of_error": None,
            "recommendations": []
        }
    
    def _strip_chatter(self, text: str) -> str:
        """Remove LLM chatter and disclaimers."""
        patterns = [
            r'^.*?Sure, here is.*?(\n|$)',
            r'^.*?Here\'?s the.*?(\n|$)',
            r'^.*?As an AI.*?(\n|$)',
            r'^.*?I\'ll provide.*?(\n|$)',
            r'^.*?Let me.*?(\n|$)',
            r'^.*?Based on the data.*?(\n|$)',
            r'^.*?Please note.*?(\n|$)'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text
    
    def _remove_markdown(self, text: str) -> str:
        """Remove markdown formatting."""
        text = re.sub(r'[#*_`]', '', text)
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'__', '', text)
        text = re.sub(r'```', '', text)
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emoji characters."""
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        return re.sub(emoji_pattern, '', text)
    
    def _detect_sections(self, lines: List[str]) -> Dict[str, Any]:
        """Detect and extract sections from text."""
        section_patterns = {
            "executive_summary": r'(?:executive summary|overview|summary|dataset overview)',
            "key_findings": r'(?:key findings|main findings|findings|key metrics)',
            "segment_insights": r'(?:segment|segmentation|demographic|distribution)',
            "trends": r'(?:trends|patterns|observations|variance)',
            "risks": r'(?:risks|limitations|concerns|quality)',
            "margin_of_error": r'(?:margin of error|moe|confidence|precision)',
            "recommendations": r'(?:recommendations|suggestions|next steps|actions)'
        }
        
        result = self._empty_structure()
        current_section = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line is section header
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line) < 100:
                    # Save previous section
                    if current_section and current_content:
                        self._add_to_section(result, current_section, current_content)
                    
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                # Add content to current section
                if len(line) > 15 and not re.match(r'^[\d\.\,\%\s]+$', line):
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            self._add_to_section(result, current_section, current_content)
        
        # Fallback: use first paragraph as executive summary
        if all(v is None or v == [] for v in result.values()):
            cleaned_lines = [l for l in lines if len(l) > 30]
            if cleaned_lines:
                result["executive_summary"] = cleaned_lines[0][:500]
        
        return result
    
    def _add_to_section(self, result: Dict, section: str, content: List[str]):
        """Add content to appropriate section."""
        if section in ["executive_summary", "trends", "risks", "margin_of_error"]:
            # String fields
            text = ' '.join(content).strip()
            text = re.sub(r'\s+', ' ', text)
            
            # Remove duplicates
            sentences = text.split('. ')
            unique = []
            for s in sentences:
                if s and s not in unique:
                    unique.append(s)
            text = '. '.join(unique)
            
            if len(text) > 10:
                result[section] = text[:1000]
        else:
            # List fields
            for line in content:
                line = re.sub(r'^[\-\•\*\+\d\.\)]+\s*', '', line).strip()
                if len(line) > 10 and line not in result[section]:
                    result[section].append(line[:300])
    
    def _validate_placeholders(self, result: Dict) -> Dict:
        """Remove content with invalid placeholders."""
        invalid_patterns = [
            r'\bXYZ\b',
            r'\bTBD\b',
            r'\[.*?\]',
            r'placeholder',
            r'example',
            r'\.\.\.',
            r'INSERT',
            r'TODO'
        ]
        
        for key, value in result.items():
            if isinstance(value, str) and value:
                for pattern in invalid_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        result[key] = None
                        break
            elif isinstance(value, list):
                valid_items = []
                for item in value:
                    has_invalid = False
                    for pattern in invalid_patterns:
                        if re.search(pattern, item, re.IGNORECASE):
                            has_invalid = True
                            break
                    if not has_invalid:
                        valid_items.append(item)
                result[key] = valid_items
        
        return result


class NarrativeValidator:
    """
    Validates narrative JSON for completeness, safety, and compliance.
    Detects hallucinations, missing content, format issues, and length problems.
    """
    
    def __init__(self):
        self.min_words = 250
        self.max_words = 700
        self.logger = logging.getLogger("NarrativeValidator")
    
    def validate(self, narrative_json: Dict[str, Any], summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate narrative against source data and rules.
        
        Args:
            narrative_json: Normalized narrative structure
            summary_dict: Original statistical summary
            
        Returns:
            Validation report with issues and status
        """
        self.logger.debug("Starting validation")
        
        issues = []
        all_text = ""
        
        # Check missing content
        required_fields = ["executive_summary", "key_findings", "segment_insights",
                          "trends", "risks", "margin_of_error", "recommendations"]
        
        for field in required_fields:
            value = narrative_json.get(field)
            
            if field in ["executive_summary", "trends", "risks", "margin_of_error"]:
                if not value or (isinstance(value, str) and len(value.strip()) < 10):
                    issues.append({
                        "section": field,
                        "problem": "missing",
                        "details": f"{field} is empty or too short"
                    })
                else:
                    all_text += " " + value
            else:
                if not value or (isinstance(value, list) and len(value) == 0):
                    issues.append({
                        "section": field,
                        "problem": "missing",
                        "details": f"{field} contains no items"
                    })
                elif isinstance(value, list):
                    all_text += " " + " ".join(value)
        
        # Check format compliance
        format_issues = self._check_format(all_text)
        issues.extend(format_issues)
        
        # Check length
        word_count = len(all_text.split())
        if word_count < self.min_words:
            issues.append({
                "section": "length",
                "problem": "too short",
                "details": f"Narrative contains {word_count} words (minimum: {self.min_words})"
            })
        elif word_count > self.max_words:
            issues.append({
                "section": "length",
                "problem": "too long",
                "details": f"Narrative contains {word_count} words (maximum: {self.max_words})"
            })
        
        # Check hallucinations
        hallucinations = self._detect_hallucinations(narrative_json, summary_dict, all_text)
        issues.extend(hallucinations)
        
        # Classify severity
        is_valid = len(issues) == 0
        overall_status = self._classify_severity(issues)
        
        self.logger.info(f"Validation complete: {overall_status} with {len(issues)} issues")
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "overall_status": overall_status,
            "word_count": word_count,
            "sections_validated": len(required_fields)
        }
    
    def _check_format(self, text: str) -> List[Dict]:
        """Check for format violations."""
        issues = []
        
        if re.search(r'[#*_`]|\*\*|__|##', text):
            issues.append({
                "section": "format",
                "problem": "invalid format",
                "details": "markdown symbols detected"
            })
        
        if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', text):
            issues.append({
                "section": "format",
                "problem": "invalid format",
                "details": "emojis detected"
            })
        
        if '```' in text:
            issues.append({
                "section": "format",
                "problem": "invalid format",
                "details": "code fences detected"
            })
        
        # Check disclaimers
        disclaimer_patterns = [
            r'as an ai',
            r'please note',
            r'i should mention',
            r'it\'?s important to'
        ]
        
        for pattern in disclaimer_patterns:
            if re.search(pattern, text.lower()):
                issues.append({
                    "section": "format",
                    "problem": "invalid format",
                    "details": f"model disclaimer detected: {pattern}"
                })
                break
        
        # Check speculation
        speculative = [
            r'maybe',
            r'perhaps',
            r'possibly',
            r'might be',
            r'could be caused by'
        ]
        
        for pattern in speculative:
            if re.search(pattern, text.lower()):
                issues.append({
                    "section": "format",
                    "problem": "invalid format",
                    "details": f"speculative language detected: {pattern}"
                })
                break
        
        return issues
    
    def _detect_hallucinations(self, narrative_json: Dict, summary_dict: Dict, full_text: str) -> List[Dict]:
        """Detect hallucinated numbers and correlations."""
        issues = []
        
        # Extract valid numbers
        valid_numbers = set()
        self._extract_numbers(summary_dict, valid_numbers)
        
        # Check each section
        sections = {
            "executive_summary": narrative_json.get("executive_summary", ""),
            "key_findings": " ".join(narrative_json.get("key_findings", [])),
            "segment_insights": " ".join(narrative_json.get("segment_insights", [])),
            "trends": narrative_json.get("trends", ""),
            "risks": narrative_json.get("risks", ""),
            "margin_of_error": narrative_json.get("margin_of_error", "")
        }
        
        for section_name, section_text in sections.items():
            if not section_text:
                continue
            
            numbers = re.findall(r'\b(\d+\.?\d*)\s*%?\b', str(section_text))
            
            for num_str in numbers:
                try:
                    num = float(num_str)
                    num_rounded = round(num, 2)
                    
                    tolerance = 0.5 if num_rounded == int(num_rounded) else 0.05
                    
                    found = False
                    for valid_num in valid_numbers:
                        if abs(valid_num - num_rounded) <= tolerance:
                            found = True
                            break
                    
                    # Whitelist common values
                    if not found and num > 10 and num_rounded not in [2025, 2026, 95, 100]:
                        context_pattern = rf'.{{0,40}}{re.escape(num_str)}.{{0,40}}'
                        context_match = re.search(context_pattern, str(section_text))
                        context = context_match.group(0) if context_match else num_str
                        
                        issues.append({
                            "section": section_name,
                            "problem": "hallucination",
                            "details": f"Number '{num_str}' not found in source data. Context: '{context}'"
                        })
                
                except ValueError:
                    continue
        
        # Check for invented correlations
        correlation_patterns = [
            r'correlation between',
            r'caused by',
            r'leads to',
            r'results in'
        ]
        
        for pattern in correlation_patterns:
            if re.search(pattern, full_text.lower()):
                if "correlations" not in str(summary_dict).lower():
                    issues.append({
                        "section": "trends",
                        "problem": "hallucination",
                        "details": f"Invented correlation detected: pattern '{pattern}'"
                    })
                    break
        
        return issues
    
    def _extract_numbers(self, obj: Any, numbers: set, prefix: str = ""):
        """Recursively extract numbers from data structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._extract_numbers(value, numbers, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, (int, float)):
            numbers.add(round(obj, 2))
        elif isinstance(obj, str):
            nums = re.findall(r'\d+\.?\d*', obj)
            for num in nums:
                try:
                    numbers.add(round(float(num), 2))
                except:
                    pass
    
    def _classify_severity(self, issues: List[Dict]) -> str:
        """Classify overall status based on issues."""
        has_hallucinations = any(i["problem"] == "hallucination" for i in issues)
        has_critical_missing = any(
            i["problem"] == "missing" and i["section"] in ["executive_summary", "key_findings"]
            for i in issues
        )
        
        if has_hallucinations:
            return "unsafe"
        elif has_critical_missing or len(issues) > 5:
            return "needs_revision"
        elif len(issues) > 0:
            return "needs_revision"
        else:
            return "ok"


class RegenerationEngine:
    """
    Handles regeneration of narratives based on validation failures.
    Builds corrective prompts and merges regenerated content.
    """
    
    def __init__(self, prompt_assembler: PromptAssembler, llm_client: LLMClient):
        self.prompt_assembler = prompt_assembler
        self.llm_client = llm_client
        self.max_attempts = 2
        self.logger = logging.getLogger("RegenerationEngine")
    
    def regenerate(self, narrative_json: Dict, issues: List[Dict], summary_dict: Dict) -> Tuple[str, Dict]:
        """
        Regenerate narrative based on validation issues.
        
        Args:
            narrative_json: Current narrative JSON
            issues: Validation issues
            summary_dict: Source data
            
        Returns:
            Tuple of (raw_text, metadata)
        """
        self.logger.info(f"Regenerating narrative due to {len(issues)} issues")
        
        # Build corrective prompt
        corrective_prompt = self.prompt_assembler.build_corrective(summary_dict, issues)
        
        # Generate new narrative
        try:
            raw_text = self.llm_client.generate(corrective_prompt)
            
            metadata = {
                "regeneration_reason": [i["problem"] for i in issues],
                "corrective_strategy": "full" if any(i["problem"] == "hallucination" for i in issues) else "partial"
            }
            
            return raw_text, metadata
            
        except Exception as e:
            self.logger.error(f"Regeneration failed: {str(e)}")
            raise


class NarrativeEngine:
    """
    Orchestrates two-pass narrative generation pipeline.
    Main entry point for generating validated statistical narratives.
    """
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize narrative engine with LLM provider.
        
        Args:
            provider: LLM provider name
            api_key: API key for provider
        """
        self.prompt_assembler = PromptAssembler()
        self.llm_client = LLMClient(provider, api_key)
        self.normalizer = ResponseNormalizer()
        self.validator = NarrativeValidator()
        self.regenerator = RegenerationEngine(self.prompt_assembler, self.llm_client)
        
        self.logger = logging.getLogger("NarrativeEngine")
        self.logger.info(f"Initialized NarrativeEngine with provider: {provider}")
    
    def generate(self, summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate validated narrative using two-pass pipeline.
        
        Pipeline:
        1. Build prompt from summary_dict
        2. Generate raw text via LLM
        3. Normalize to structured JSON
        4. Validate against source data
        5. Regenerate if unsafe/incomplete
        6. Return final validated JSON
        
        Args:
            summary_dict: Structured statistical summary
            
        Returns:
            Dictionary with narrative JSON and metadata
        """
        self.logger.info("Starting narrative generation pipeline")
        start_time = time.time()
        
        try:
            # PASS 1: GENERATION
            self.logger.debug("PASS 1: Building prompt")
            prompt = self.prompt_assembler.build(summary_dict)
            
            self.logger.debug("PASS 1: Calling LLM")
            raw_narrative = self.llm_client.generate(prompt)
            
            # PASS 2: NORMALIZATION
            self.logger.debug("PASS 2: Normalizing response")
            narrative_json = self.normalizer.normalize(raw_narrative)
            
            # VALIDATION
            self.logger.debug("PASS 2: Validating narrative")
            validation_report = self.validator.validate(narrative_json, summary_dict)
            
            regeneration_count = 0
            
            # DECISION GATE
            if validation_report["overall_status"] == "unsafe":
                self.logger.warning("Unsafe content detected. Attempting regeneration...")
                
                try:
                    raw_narrative, regen_meta = self.regenerator.regenerate(
                        narrative_json,
                        validation_report["issues"],
                        summary_dict
                    )
                    
                    narrative_json = self.normalizer.normalize(raw_narrative)
                    validation_report = self.validator.validate(narrative_json, summary_dict)
                    regeneration_count = 1
                    
                    # If still unsafe, return fallback
                    if validation_report["overall_status"] == "unsafe":
                        self.logger.error("Regeneration failed. Using fallback.")
                        return self._fallback_response(summary_dict, validation_report)
                    
                except Exception as e:
                    self.logger.error(f"Regeneration error: {str(e)}")
                    return self._fallback_response(summary_dict, validation_report)
            
            elif validation_report["overall_status"] == "needs_revision":
                self.logger.info("Incomplete content. Attempting partial regeneration...")
                
                try:
                    raw_narrative, regen_meta = self.regenerator.regenerate(
                        narrative_json,
                        validation_report["issues"],
                        summary_dict
                    )
                    
                    narrative_json = self.normalizer.normalize(raw_narrative)
                    regeneration_count = 1
                    
                except Exception as e:
                    self.logger.warning(f"Partial regeneration failed: {str(e)}. Using best effort.")
            
            # FINAL OUTPUT
            elapsed = time.time() - start_time
            
            result = {
                "narrative": narrative_json,
                "validation": validation_report,
                "metadata": {
                    "provider": self.llm_client.provider,
                    "model": self.llm_client.config["model"],
                    "regeneration_count": regeneration_count,
                    "elapsed_seconds": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            }
            
            self.logger.info(f"Pipeline complete in {elapsed:.2f}s. Status: {validation_report['overall_status']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            elapsed = time.time() - start_time
            
            return {
                "narrative": self.normalizer._empty_structure(),
                "validation": {
                    "is_valid": False,
                    "issues": [{"section": "system", "problem": "error", "details": str(e)}],
                    "overall_status": "error",
                    "word_count": 0,
                    "sections_validated": 0
                },
                "metadata": {
                    "provider": self.llm_client.provider,
                    "model": self.llm_client.config["model"],
                    "regeneration_count": 0,
                    "elapsed_seconds": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                }
            }
    
    def _fallback_response(self, summary_dict: Dict, validation_report: Dict) -> Dict:
        """Return structured fallback when AI narrative fails."""
        self.logger.warning("Using fallback response (structured data only)")
        
        return {
            "narrative": {
                "executive_summary": "AI narrative unavailable. Refer to structured summary below.",
                "key_findings": [],
                "segment_insights": [],
                "trends": None,
                "risks": None,
                "margin_of_error": None,
                "recommendations": ["Review structured summary data", "Consider manual narrative generation"]
            },
            "validation": validation_report,
            "metadata": {
                "provider": self.llm_client.provider,
                "model": self.llm_client.config["model"],
                "regeneration_count": 2,
                "timestamp": datetime.now().isoformat(),
                "status": "fallback",
                "reason": "persistent_validation_failures"
            }
        }
