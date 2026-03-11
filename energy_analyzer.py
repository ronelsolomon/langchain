#!/usr/bin/env python3
"""
Energy Data Analyzer with LangChain
==================================
Uses LangChain to analyze and query energy data collected from EIA, OpenEI, and NREL APIs.
Features:
- Natural language queries about energy data
- Data summarization and insights
- Rate comparison analysis
- Energy trend analysis
"""

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load API keys (for potential OpenAI usage)
def load_api_keys():
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

class EnergyDataAnalyzer:
    def __init__(self, data_dir: str = "./energy_data_output"):
        self.data_dir = Path(data_dir)
        self.api_keys = load_api_keys()
        self.documents = []
        self.vector_store = None
        self.llm = None
        self.embeddings = None
        
        # Initialize LLM based on available services
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM and embeddings based on available services"""
        try:
            # Try Ollama first (local, free)
            self.llm = Ollama(model="llama3:latest")
            self.embeddings = OllamaEmbeddings(model="llama3:latest")
            print("✅ Using Ollama (Llama3:latest) for local processing")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            
            # Fallback to OpenAI if API key is available
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
                    print("Please install Ollama or add OPENAI_API_KEY to env file")
                    sys.exit(1)
            else:
                print("❌ No LLM available. Install Ollama or add OPENAI_API_KEY to env file")
                sys.exit(1)
    
    def load_energy_data(self):
        """Load all energy data files and create documents"""
        print("📊 Loading energy data...")
        
        # Load EIA electricity prices
        prices_file = self.data_dir / "eia_electricity_prices_commercial.csv"
        if prices_file.exists():
            df = pd.read_csv(prices_file)
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
        
        # Load EIA generation data
        generation_file = self.data_dir / "eia_electricity_generation.csv"
        if generation_file.exists():
            df = pd.read_csv(generation_file)
            doc_content = f"""
            EIA Electricity Generation Data:
            - Total records: {len(df)}
            - Date range: {df['period'].min()} to {df['period'].max()}
            - States covered: {df['stateid'].nunique()}
            - Fuel types: {df['fueltypeid'].nunique()} different types
            
            Generation by fuel type (total):
            {df.groupby('fueltypeid')['generation'].sum().sort_values(ascending=False).head(10).to_string()}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "eia_generation", "type": "summary"}
            ))
        
        # Load OpenEI utility rates
        rates_file = self.data_dir / "openei_utility_rates_flat.csv"
        if rates_file.exists():
            df = pd.read_csv(rates_file)
            doc_content = f"""
            OpenEI Utility Rate Database:
            - Total rate structures: {len(df)}
            - States covered: {df['state'].nunique()}
            - Utilities: {df['utility_name'].nunique()}
            - Rates with TOU (Time of Use): {df['has_tou'].sum()}
            - Rates with demand charges: {df['has_demand_charges'].sum()}
            
            Average fixed monthly charges: ${df['fixed_monthly_charge'].mean():.2f}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "openei_rates", "type": "summary"}
            ))
        
        # Load NREL utility rates by zip
        nrel_file = self.data_dir / "nrel_utility_rates_by_zip.csv"
        if nrel_file.exists():
            df = pd.read_csv(nrel_file)
            doc_content = f"""
            NREL Utility Rates by Location:
            - Locations analyzed: {len(df)}
            - Utilities: {df['utility_name'].nunique()}
            
            Residential rate statistics:
            - Average: ${df['residential_rate_usd_per_kwh'].mean():.4f}/kWh
            - Min: ${df['residential_rate_usd_per_kwh'].min():.4f}/kWh
            - Max: ${df['residential_rate_usd_per_kwh'].max():.4f}/kWh
            
            Top 5 most expensive locations:
            {df.nlargest(5, 'residential_rate_usd_per_kwh')[['zip_code', 'utility_name', 'residential_rate_usd_per_kwh']].to_string()}
            
            Top 5 cheapest locations:
            {df.nsmallest(5, 'residential_rate_usd_per_kwh')[['zip_code', 'utility_name', 'residential_rate_usd_per_kwh']].to_string()}
            """
            self.documents.append(Document(
                page_content=doc_content,
                metadata={"source": "nrel_rates", "type": "summary"}
            ))
        
        print(f"✅ Loaded {len(self.documents)} data summaries")
        
    def create_vector_store(self):
        """Create vector store for semantic search"""
        if not self.documents:
            print("❌ No documents loaded. Call load_energy_data() first.")
            return
        
        print("🔍 Creating vector store for semantic search...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(self.documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(f"✅ Created vector store with {len(texts)} chunks")
    
    def query_data(self, question: str) -> str:
        """Query the energy data using natural language"""
        if not self.vector_store:
            print("❌ Vector store not created. Call create_vector_store() first.")
            return ""
        
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        
        # Create prompt
        template = """
        You are an energy data analyst. Based on the following energy data information, 
        please answer the user's question accurately and concisely.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = chain.invoke(question)
            return result
        except Exception as e:
            return f"Error processing query: {e}"
    
    def generate_insights(self) -> str:
        """Generate general insights about the energy data"""
        if not self.documents:
            return "No data available for analysis."
        
        context = "\n\n".join([doc.page_content for doc in self.documents])
        
        template = """
        Based on the following energy data, provide 5 key insights about electricity pricing,
        generation trends, and utility rates in the United States. Focus on actionable
        business intelligence and interesting patterns.
        
        Data Summary:
        {context}
        
        Key Insights:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({"context": context})
        except Exception as e:
            return f"Error generating insights: {e}"
    
    def compare_rates(self, zip_codes: List[str]) -> str:
        """Compare utility rates across different zip codes"""
        nrel_file = self.data_dir / "nrel_utility_rates_by_zip.csv"
        if not nrel_file.exists():
            return "NREL rate data not available."
        
        df = pd.read_csv(nrel_file)
        filtered_df = df[df['zip_code'].isin(zip_codes)]
        
        if filtered_df.empty:
            return f"No data found for zip codes: {zip_codes}"
        
        comparison = filtered_df[['zip_code', 'utility_name', 'residential_rate_usd_per_kwh', 
                                 'commercial_rate_usd_per_kwh', 'industrial_rate_usd_per_kwh']].to_string()
        
        template = """
        Analyze this utility rate comparison and provide insights:
        
        {comparison}
        
        Provide analysis on:
        1. Which locations have the highest/lowest rates
        2. Rate differences between residential, commercial, and industrial
        3. Any notable patterns or anomalies
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["comparison"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({"comparison": comparison})
        except Exception as e:
            return f"Error comparing rates: {e}"


def main():
    print("🚀 Energy Data Analyzer with LangChain")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EnergyDataAnalyzer()
    
    # Load data
    analyzer.load_energy_data()
    
    # Create vector store
    analyzer.create_vector_store()
    
    print("\n🎯 Interactive Query Mode")
    print("Type 'quit' to exit, 'insights' for general analysis")
    print("Example queries:")
    print("- Which states have the highest electricity prices?")
    print("- What are the average residential rates?")
    print("- Compare rates in California vs Texas")
    print("- Which utilities have time-of-use pricing?")
    
    while True:
        try:
            question = input("\n🔍 Ask about energy data: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'insights':
                print("\n💡 Generating insights...")
                result = analyzer.generate_insights()
                print(result)
            elif question.lower().startswith('compare'):
                # Extract zip codes from compare command
                zip_codes = question.split()[1:]
                if len(zip_codes) >= 2:
                    print(f"\n📊 Comparing rates for: {zip_codes}")
                    result = analyzer.compare_rates(zip_codes)
                    print(result)
                else:
                    print("Usage: compare <zip1> <zip2> [zip3...]")
            else:
                print(f"\n🤔 Analyzing: {question}")
                result = analyzer.query_data(question)
                print(result)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
