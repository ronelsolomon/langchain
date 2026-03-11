#!/usr/bin/env python3
"""
Pattern Matcher for Energy Data Analysis
====================================
Integrates regex pattern matching with LangChain for advanced data filtering and validation.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document

class EnergyPatternMatcher:
    """Advanced pattern matching for energy data analysis"""
    
    def __init__(self):
        self.patterns = {
            # Utility name patterns
            'utility_company': r'.*(Power|Energy|Electric|Light|Utility).*',
            'investor_owned': r'.*(Investor|Investment|Public).*',
            'municipal': r'.*(City|Municipal|County|Public).*',
            'cooperative': r'.*(Cooperative|Coop|Rural).*',
            
            # Rate structure patterns
            'time_of_use': r'.*(TOU|Time of Use|Peak|Off.*Peak).*',
            'tiered_rate': r'.*(Tier|Block|Step).*',
            'demand_charge': r'.*(Demand|KW|kW).*',
            'fixed_charge': r'.*(Fixed|Monthly|Base).*',
            
            # Energy source patterns
            'renewable': r'.*(Solar|Wind|Hydro|Geothermal|Biomass).*',
            'fossil_fuel': r'.*(Coal|Natural.*Gas|Oil|Petroleum|Nuclear).*',
            
            # Location patterns
            'zip_code': r'\d{5}(-\d{4})?',
            'state_code': r'\b[A-Z]{2}\b',
            'phone_number': r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',
            
            # Data patterns
            'price_per_kwh': r'\$?\d*\.?\d+\s*(cents?|\/kWh|cents?\/kWh)',
            'percentage': r'\d+\.?\d*%',
            'large_number': r'\d{1,3}(,\d{3})*(\.\d+)?',
        }
    
    def is_match(self, s: str, p: str) -> bool:
        """
        Original pattern matching function with improvements
        Returns True if the entire string matches the pattern
        """
        try:
            a = re.findall(p, s)
            if len(a) > 0:
                # For exact match, compare the first match to the original string
                return a[0] == s
            else:
                return False
        except re.error as e:
            print(f"Regex error: {e}")
            return False
    
    def contains_pattern(self, text: str, pattern_name: str) -> bool:
        """
        Check if text contains a predefined pattern
        """
        if pattern_name not in self.patterns:
            return False
        
        pattern = self.patterns[pattern_name]
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def extract_matches(self, text: str, pattern_name: str) -> List[str]:
        """
        Extract all matches of a predefined pattern from text
        """
        if pattern_name not in self.patterns:
            return []
        
        pattern = self.patterns[pattern_name]
        return re.findall(pattern, text, re.IGNORECASE)
    
    def filter_dataframe(self, df: pd.DataFrame, column: str, pattern_name: str) -> pd.DataFrame:
        """
        Filter DataFrame rows based on pattern matching in a specific column
        """
        if pattern_name not in self.patterns:
            return df
        
        pattern = self.patterns[pattern_name]
        mask = df[column].str.contains(pattern, case=False, na=False)
        return df[mask]
    
    def analyze_utility_rates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze utility rates using pattern matching
        """
        analysis = {
            'total_utilities': len(df),
            'utility_types': {},
            'rate_structures': {},
            'energy_sources': {}
        }
        
        # Analyze utility types
        for util_type in ['investor_owned', 'municipal', 'cooperative']:
            mask = df['utility_name'].str.contains(
                self.patterns[util_type], case=False, na=False
            )
            analysis['utility_types'][util_type] = mask.sum()
        
        # Analyze rate structures
        for rate_type in ['time_of_use', 'tiered_rate', 'demand_charge']:
            if 'rate_name' in df.columns:
                mask = df['rate_name'].str.contains(
                    self.patterns[rate_type], case=False, na=False
                )
                analysis['rate_structures'][rate_type] = mask.sum()
        
        return analysis


# LangChain Tools Integration
@tool
def match_energy_pattern(text: str, pattern_name: str) -> str:
    """
    Match energy data patterns using regex.
    
    Args:
        text: Text to search in
        pattern_name: Name of pattern (utility_company, time_of_use, renewable, etc.)
    
    Returns:
        Description of matches found
    """
    matcher = EnergyPatternMatcher()
    
    if pattern_name not in matcher.patterns:
        return f"Pattern '{pattern_name}' not found. Available patterns: {list(matcher.patterns.keys())}"
    
    matches = matcher.extract_matches(text, pattern_name)
    
    if matches:
        return f"Found {len(matches)} matches for '{pattern_name}' pattern: {matches}"
    else:
        return f"No matches found for '{pattern_name}' pattern in the text."


@tool
def analyze_utility_type(utility_name: str) -> str:
    """
    Analyze utility company type using pattern matching.
    
    Args:
        utility_name: Name of the utility company
    
    Returns:
        Analysis of utility type and characteristics
    """
    matcher = EnergyPatternMatcher()
    
    analysis = []
    
    # Check utility type
    if matcher.contains_pattern(utility_name, 'investor_owned'):
        analysis.append("Investor-owned utility")
    elif matcher.contains_pattern(utility_name, 'municipal'):
        analysis.append("Municipal utility")
    elif matcher.contains_pattern(utility_name, 'cooperative'):
        analysis.append("Cooperative utility")
    else:
        analysis.append("Unknown utility type")
    
    # Check for other characteristics
    if matcher.contains_pattern(utility_name, 'renewable'):
        analysis.append("May focus on renewable energy")
    
    return f"Utility '{utility_name}': {', '.join(analysis)}"


@tool
def extract_price_info(rate_text: str) -> str:
    """
    Extract price information from rate text using regex patterns.
    
    Args:
        rate_text: Text containing rate information
    
    Returns:
        Extracted price information
    """
    matcher = EnergyPatternMatcher()
    
    prices = matcher.extract_matches(rate_text, 'price_per_kwh')
    percentages = matcher.extract_matches(rate_text, 'percentage')
    
    result = []
    if prices:
        result.append(f"Prices found: {prices}")
    if percentages:
        result.append(f"Percentages found: {percentages}")
    
    if not result:
        return "No price information found in the text."
    
    return "; ".join(result)


# Integration with EnergyDataAnalyzer
def enhance_analyzer_with_patterns(analyzer):
    """
    Add pattern matching capabilities to existing EnergyDataAnalyzer
    """
    matcher = EnergyPatternMatcher()
    
    # Add pattern matching methods to the analyzer
    analyzer.pattern_matcher = matcher
    
    def enhanced_query(question: str) -> str:
        """Enhanced query that includes pattern matching"""
        # Check if question is about pattern matching
        if any(keyword in question.lower() for keyword in ['match', 'pattern', 'extract', 'find']):
            # Extract pattern name from question
            for pattern_name in matcher.patterns.keys():
                if pattern_name.replace('_', ' ') in question.lower():
                    return match_energy_pattern(question, pattern_name)
        
        # Use original query method
        return analyzer.query_data(question)
    
    analyzer.enhanced_query = enhanced_query
    
    def pattern_analysis_report() -> str:
        """Generate pattern-based analysis report"""
        if not hasattr(analyzer, 'dataframes'):
            return "No data available for pattern analysis."
        
        reports = []
        
        # Analyze utility rates if available
        if 'openei_rates' in analyzer.dataframes:
            analysis = matcher.analyze_utility_rates(analyzer.dataframes['openei_rates'])
            reports.append("Utility Rate Analysis:")
            reports.append(f"  Total utilities: {analysis['total_utilities']}")
            reports.append("  Utility types:")
            for util_type, count in analysis['utility_types'].items():
                reports.append(f"    {util_type}: {count}")
        
        return "\n".join(reports)
    
    analyzer.pattern_analysis_report = pattern_analysis_report
    
    return analyzer


if __name__ == "__main__":
    # Demo the pattern matcher
    matcher = EnergyPatternMatcher()
    
    print("🔍 Energy Pattern Matcher Demo")
    print("=" * 40)
    
    # Test the original function
    test_cases = [
        ("Pacific Gas & Electric Co", r'.*(Gas|Electric).*'),
        ("12345", r'\d{5}'),
        ("Hello World", r'.*World.*'),
        ("Mismatch", r'.*Electric.*')
    ]
    
    print("\n📋 Testing is_match function:")
    for text, pattern in test_cases:
        result = matcher.is_match(text, pattern)
        print(f"  '{text}' ~ '{pattern}' -> {result}")
    
    # Test pattern matching
    print("\n🎯 Testing predefined patterns:")
    test_texts = [
        "Pacific Gas & Electric Company offers time-of-use rates",
        "Los Angeles Department of Water and Power",
        "Solar Renewable Energy Credits at 15% discount",
        "Rate: $0.15/kWh for residential customers"
    ]
    
    for text in test_texts:
        print(f"\n  Text: '{text}'")
        for pattern_name in ['utility_company', 'time_of_use', 'renewable', 'price_per_kwh']:
            if matcher.contains_pattern(text, pattern_name):
                matches = matcher.extract_matches(text, pattern_name)
                print(f"    ✅ {pattern_name}: {matches}")
    
    print("\n🔧 LangChain tools ready for integration!")
