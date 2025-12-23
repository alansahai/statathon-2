import pandas as pd
import numpy as np

class AIEngine:
    """
    Generates Natural Language Summaries and Statistical Insights 
    to fulfill 'Automated Narrative' criteria.
    """

    @staticmethod
    def generate_executive_summary(filename, stats_results, metadata):
        """
        Constructs a narrative executive summary based on the calculated estimates.
        """
        if not stats_results:
            return "No statistical analysis has been performed yet."

        narrative = []
        
        # 1. Introduction
        narrative.append(f"<b>Executive Summary for {filename}</b>")
        narrative.append(f"This report presents the findings from the survey data analysis. "
                         f"A total of {len(stats_results)} variables were analyzed using design-based estimation methods.")
        
        # 2. Variable-wise Insights
        for stat in stats_results:
            col = stat['column']
            mean = stat['mean']
            se = stat['std_error']
            n_eff = stat['n_effective']
            
            # Analyze Precision
            cv = (se / mean) * 100 if mean != 0 else 0
            reliability = "high" if cv < 5 else "moderate" if cv < 15 else "low"
            
            paragraph = (
                f"For the variable '<b>{col}</b>', the estimated weighted mean is <b>{mean}</b> "
                f"(Standard Error: {se}). Based on the effective sample size of {n_eff}, "
                f"the estimate is considered to have <b>{reliability} statistical reliability</b> "
                f"(CV: {cv:.1f}%)."
            )
            
            # Add Population Context if available
            if stat.get('population_estimate') != "N/A":
                paragraph += (
                    f" When projected to the total population, this corresponds to an estimated total of "
                    f"<b>{stat['population_estimate']}</b>."
                )
                
            narrative.append(paragraph)

        # 3. Methodology Note
        narrative.append(
            "<br><i>Methodology Note: Estimates were calculated using Kish's Effective Sample Size "
            "weighting to correct for design effects. 95% Confidence Intervals were computed using "
            "the Student's t-distribution.</i>"
        )
        
        return "<br><br>".join(narrative)