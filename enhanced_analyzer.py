#!/usr/bin/env python3
"""
Enhanced Energy Data Analyzer with Pattern Matching
==============================================
Integrates LangChain with regex pattern matching for advanced energy data analysis.
Includes your isMatch function and additional pattern matching capabilities.
"""

import os
import re
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Import pattern matcher
from pattern_matcher import EnergyPatternMatcher, enhance_analyzer_with_patterns

class EnhancedEnergyAnalyzer:
    """Enhanced analyzer with pattern matching capabilities"""
    
    def __init__(self, data_dir: str = "./energy_data_output"):
        self.data_dir = Path(data_dir)
        self.api_keys = self.load_api_keys()
        self.documents = []
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        self.pattern_matcher = EnergyPatternMatcher()
        self.dataframes = {}
        
        # Initialize LLM
        self._initialize_llm()
        
        # Load data
        self.load_energy_data()
        
        # Create vector store
        self.create_vector_store()
    
    def load_api_keys(self):
        """Load API keys from env file"""
        keys = {}
        try:
            with open("env") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        keys[key.strip()] = value.strip().strip('"\'')
        except FileNotFoundError:
            print("Warning: env file not found")
        return keys
    
    def _initialize_llm(self):
        """Initialize LLM and embeddings"""
        try:
            # Try Ollama first
            self.llm = Ollama(model="llama3:latest")
            self.embeddings = OllamaEmbeddings(model="llama3:latest")
            print("✅ Using Ollama (Llama3:latest) for local processing")
        except Exception as e:
            # Try with llama3.2 as fallback
            try:
                self.llm = Ollama(model="llama3:latest")
                self.embeddings = OllamaEmbeddings(model="llama3:latest")
                print("✅ Using Ollama (Llama3:latest) for local processing")
            except Exception as e2:
                print(f"⚠️  Ollama not available: {e}")
                print("💡 Tip: Make sure Ollama is running and a model is pulled with: ollama pull llama3:latest")
            
            if self.api_keys.get("OPENAI_API_KEY"):
                try:
                    self.llm = ChatOpenAI(
                        api_key=self.api_keys["OPENAI_API_KEY"],
                        model="gpt-3.5-turbo"
                    )
                    self.embeddings = OpenAIEmbeddings(
                        api_key=self.api_keys["OPENAI_API_KEY"]
                    )
                    print("✅ Using OpenAI GPT-3.5-turbo")
                except Exception as e:
                    print(f"❌ OpenAI initialization failed: {e}")
                    sys.exit(1)
            else:
                print("❌ No LLM available")
                sys.exit(1)
    
    # Your original pattern matching function
    def isMatch(self, s: str, p: str) -> bool:
        """
        Original pattern matching function
        Returns True if the entire string matches the pattern
        """
        a = re.findall(p, s)
        print(f"Pattern '{p}' matches in '{s}': {a}")
        if len(a) > 0:
            if a[0] == s:
                return True
            else:
                return False
        else:
            return False
    
    def load_energy_data(self):
        """Load energy data and create documents"""
        print("📊 Loading energy data...")
        
        # Load EIA electricity prices
        prices_file = self.data_dir / "eia_electricity_prices_commercial.csv"
        if prices_file.exists():
            df = pd.read_csv(prices_file)
            self.dataframes['eia_prices'] = df
            
            doc_content = f"""
            EIA Commercial Electricity Prices Data:
            - Total records: {len(df)}
            - Date range: {df['month'].min()} to {df['month'].max()}
            - States covered: {df['state'].nunique()}
            - Average price: {df['price_cents_per_kwh'].mean():.2f} cents/kWh
            - Price range: {df['price_cents_per_kwh'].min():.2f} - {df['price_cents_per_kwh'].max():.2f} cents/kWh
            
            Top 5 most expensive states (average):
            {df.groupby('state_name')['price_cents_per_kwh'].mean().sort_values(ascending=False).head().to_string()}
            
            Top 5 cheapest states (average):
            {df.groupby('state_name')['price_cents_per_kwh'].mean().sort_values().head().to_string()}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "eia_prices", "type": "summary"}
            ))
        
        # Load OpenEI utility rates
        rates_file = self.data_dir / "openei_utility_rates_flat.csv"
        if rates_file.exists():
            df = pd.read_csv(rates_file)
            self.dataframes['openei_rates'] = df
            
            doc_content = f"""
            OpenEI Utility Rate Database:
            - Total rate structures: {len(df)}
            - States covered: {df['state'].nunique()}
            - Utilities: {df['utility_name'].nunique()}
            - Rates with TOU (Time of Use): {df['has_tou'].sum()}
            - Rates with demand charges: {df['has_demand_charges'].sum()}
            
            Average fixed monthly charges: ${df['fixed_monthly_charge'].mean():.2f}
            
            Sample utility names:
            {df['utility_name'].head(10).to_string()}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "openei_rates", "type": "summary"}
            ))
        
        # Load NREL utility rates
        nrel_file = self.data_dir / "nrel_utility_rates_by_zip.csv"
        if nrel_file.exists():
            df = pd.read_csv(nrel_file)
            self.dataframes['nrel_rates'] = df
            
            doc_content = f"""
            NREL Utility Rates by Location:
            - Locations analyzed: {len(df)}
            - Utilities: {df['utility_name'].nunique()}
            
            Residential rate statistics:
            - Average: ${df['residential_rate_usd_per_kwh'].mean():.4f}/kWh
            - Min: ${df['residential_rate_usd_per_kwh'].min():.4f}/kWh
            - Max: ${df['residential_rate_usd_per_kwh'].max():.4f}/kWh
            
            Sample utilities and rates:
            {df[['zip_code', 'utility_name', 'residential_rate_usd_per_kwh']].head(10).to_string()}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "nrel_rates", "type": "summary"}
            ))
        
        print(f"✅ Loaded {len(self.documents)} data summaries")
    
    def create_vector_store(self):
        """Create vector store for semantic search with caching"""
        if not self.documents:
            print("❌ No documents loaded")
            return
        
        # Check for cached vector store
        cache_file = Path("./vector_store_cache.faiss")
        metadata_file = Path("./vector_store_metadata.pkl")
        
        if cache_file.exists() and metadata_file.exists():
            try:
                import pickle
                print("📂 Loading cached vector store...")
                self.vector_store = FAISS.load_local("./", self.embeddings, allow_dangerous_deserialization=True)
                with open(metadata_file, 'rb') as f:
                    cached_docs = pickle.load(f)
                
                # Check if data has changed
                current_doc_count = len(self.documents)
                if cached_docs['doc_count'] == current_doc_count:
                    print(f"✅ Loaded cached vector store ({cached_docs['chunk_count']} chunks)")
                    return
                else:
                    print("� Data changed, recreating vector store...")
            except Exception as e:
                print(f"⚠️  Cache loading failed: {e}, recreating...")
        
        print("�🔍 Creating vector store...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(self.documents)
        
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Save to cache
        try:
            import pickle
            self.vector_store.save_local("./")
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'doc_count': len(self.documents),
                    'chunk_count': len(texts),
                    'created_at': datetime.now().isoformat()
                }, f)
            print(f"✅ Created and cached vector store with {len(texts)} chunks")
        except Exception as e:
            print(f"⚠️  Failed to cache vector store: {e}")
            print(f"✅ Created vector store with {len(texts)} chunks")
    
    # LangChain tools for pattern matching
    @tool
    def match_pattern(text: str, pattern: str) -> str:
        """
        Use regex pattern matching to find matches in text.
        
        Args:
            text: Text to search in
            pattern: Regex pattern to match
        
        Returns:
            Description of matches found
        """
        matcher = EnhancedEnergyAnalyzer(None)  # Create instance for method access
        matches = re.findall(pattern, text)
        
        if matches:
            return f"Pattern '{pattern}' found {len(matches)} matches: {matches}"
        else:
            return f"No matches found for pattern '{pattern}'"
    
    @tool
    def analyze_utility_patterns(utility_name: str) -> str:
        """
        Analyze utility company name using predefined patterns.
        
        Args:
            utility_name: Name of the utility company
        
        Returns:
            Analysis of utility type and characteristics
        """
        matcher = EnergyPatternMatcher()
        analysis = []
        
        if matcher.contains_pattern(utility_name, 'investor_owned'):
            analysis.append("Investor-owned utility")
        elif matcher.contains_pattern(utility_name, 'municipal'):
            analysis.append("Municipal utility")
        elif matcher.contains_pattern(utility_name, 'cooperative'):
            analysis.append("Cooperative utility")
        
        if matcher.contains_pattern(utility_name, 'renewable'):
            analysis.append("Renewable energy focus")
        
        return f"Utility '{utility_name}': {', '.join(analysis) if analysis else 'General utility'}"
    
    def query_with_patterns(self, question: str) -> str:
        """Enhanced query that combines semantic search with pattern matching"""
        
        # Check if question involves pattern matching
        pattern_keywords = ['match', 'pattern', 'regex', 'find', 'extract', 'contains']
        if any(keyword in question.lower() for keyword in pattern_keywords):
            return self._handle_pattern_query(question)
        
        # Use standard semantic search
        return self._standard_query(question)
    
    def _handle_pattern_query(self, question: str) -> str:
        """Handle queries specifically about pattern matching"""
        
        # Test your isMatch function
        if "ismatch" in question.lower() or "exact match" in question.lower():
            return """
            The isMatch function tests if a string exactly matches a regex pattern.
            
            Example usage:
            - isMatch("12345", r"\\d{5}") -> True
            - isMatch("Hello World", r".*World.*") -> False (partial match, not exact)
            - isMatch("Pacific Gas", r".*(Gas|Electric).*") -> False (partial match)
            
            The function only returns True for complete matches, not partial matches.
            """
        
        # Handle utility pattern analysis
        if "utility" in question.lower() and "type" in question.lower():
            if 'openei_rates' in self.dataframes:
                df = self.dataframes['openei_rates']
                sample_utilities = df['utility_name'].head(5).tolist()
                
                analysis = "Utility type analysis:\n"
                for utility in sample_utilities:
                    util_type = self.pattern_matcher.analyze_utility_type(utility)
                    analysis += f"- {utility}: {util_type}\n"
                
                return analysis
        
        return "Use 'match <text> <pattern>' or ask about utility types for pattern analysis."
    
    def _standard_query(self, question: str) -> str:
        """Standard semantic search query"""
        if not self.vector_store:
            return "Vector store not available"
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        template = """
        Based on the energy data, answer: {question}
        
        Context: {context}
        
        Provide a concise, accurate answer.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            return chain.invoke(question)
        except Exception as e:
            return f"Error: {e}"
    
    def demo_pattern_matching(self):
        """Demonstrate pattern matching capabilities"""
        print("\n🎯 Pattern Matching Demo")
        print("=" * 30)
        
        # Test your isMatch function
        print("\n📋 Testing isMatch function:")
        test_cases = [
            ("Pacific Gas & Electric", r".*(Gas|Electric).*"),
            ("12345", r"\d{5}"),
            ("Hello World", r".*World.*"),
            ("$0.15/kWh", r"\$?\d*\.?\d+\s*(cents?|\/kWh)")
        ]
        
        for text, pattern in test_cases:
            result = self.isMatch(text, pattern)
            print(f"  isMatch('{text}', '{pattern}') -> {result}")
        
        # Test predefined patterns
        print("\n🔍 Testing predefined patterns:")
        if 'openei_rates' in self.dataframes:
            df = self.dataframes['openei_rates']
            sample_utilities = df['utility_name'].head(5).tolist()
            
            for utility in sample_utilities:
                print(f"\n  Utility: {utility}")
                for pattern_name in ['utility_company', 'investor_owned', 'municipal']:
                    if self.pattern_matcher.contains_pattern(utility, pattern_name):
                        print(f"    ✅ {pattern_name}")
        
        print("\n🔧 Pattern matching tools integrated with LangChain!")


def main():
    print("🚀 Enhanced Energy Data Analyzer with Pattern Matching")
    print("=" * 60)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedEnergyAnalyzer()
    
    # Run demo
    analyzer.demo_pattern_matching()
    
    print("\n🎯 Interactive Query Mode")
    print("Commands:")
    print("- Type questions about energy data")
    print("- Use 'match <text> <pattern>' for regex matching")
    print("- Ask about 'utility types' for pattern analysis")
    print("- Type 'quit' to exit")
    
    while True:
        try:
            question = input("\n🔍 Query: ").strip()
            
            if question.lower() == 'quit':
                break
            
            result = analyzer.query_with_patterns(question)
            print(f"\n🤖 {result}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
