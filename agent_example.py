#!/usr/bin/env python3
"""
Energy Data Agent Example
=========================
Shows how to create and use an agent with energy data tools.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


checkpointer = InMemorySaver()

# Energy data tools
@tool
def get_electricity_prices(state: str = None, sector: str = "commercial") -> str:
    """Get electricity prices by state and sector.
    
    Args:
        state: Two-letter state code (e.g., 'CA', 'NY')
        sector: 'commercial', 'residential', or 'industrial'
    """
    try:
        prices_file = Path("energy_data_output/eia_electricity_prices_commercial.csv")
        if not prices_file.exists():
            return "Electricity prices data not found. Please run the data collection first."
        
        df = pd.read_csv(prices_file)
        
        if state:
            state_data = df[df['state'] == state.upper()]
            if state_data.empty:
                return f"No data found for state: {state}"
            
            latest_price = state_data.iloc[-1]['price']
            return f"Latest electricity price for {state.upper()} ({sector}): ${latest_price:.4f} per kWh"
        else:
            avg_price = df['price'].mean()
            return f"Average electricity price across all states ({sector}): ${avg_price:.4f} per kWh"
            
    except Exception as e:
        return f"Error retrieving electricity prices: {str(e)}"
    
    

@tool
def get_energy_generation(source: str = None, state: str = None) -> str:
    """Get energy generation data by source and state.
    
    Args:
        source: Energy source (e.g., 'coal', 'natural_gas', 'solar', 'wind')
        state: Two-letter state code
    """
    try:
        generation_file = Path("energy_data_output/eia_electricity_generation.csv")
        if not generation_file.exists():
            return "Energy generation data not found. Please run the data collection first."
        
        df = pd.read_csv(generation_file)
        
        if source:
            source_data = df[df['energy-source'].str.lower() == source.lower()]
            if source_data.empty:
                return f"No data found for energy source: {source}"
            
            if state:
                state_data = source_data[source_data['state'] == state.upper()]
                if state_data.empty:
                    return f"No {source} generation data found for {state}"
                
                latest_gen = state_data.iloc[-1]['generation-mwh']
                return f"Latest {source} generation in {state.upper()}: {latest_gen:,.0f} MWh"
            else:
                total_gen = source_data['generation-mwh'].sum()
                return f"Total {source} generation across all states: {total_gen:,.0f} MWh"
        else:
            total_generation = df['generation-mwh'].sum()
            return f"Total electricity generation across all sources: {total_generation:,.0f} MWh"
            
    except Exception as e:
        return f"Error retrieving energy generation data: {str(e)}"

@tool
def get_utility_rates(zip_code: str = None) -> str:
    """Get utility rates by ZIP code.
    
    Args:
        zip_code: 5-digit ZIP code
    """
    try:
        rates_file = Path("energy_data_output/nrel_utility_rates_by_zip.csv")
        if not rates_file.exists():
            return "Utility rates data not found. Please run the data collection first."
        
        df = pd.read_csv(rates_file)
        
        if zip_code:
            zip_data = df[df['zip'] == int(zip_code)]
            if zip_data.empty:
                return f"No utility rate data found for ZIP code: {zip_code}"
            
            rate_info = zip_data.iloc[0]
            return f"Utility rate for {zip_code}: ${rate_info['rate']:.4f} per kWh (Utility: {rate_info.get('utility', 'Unknown')})"
        else:
            avg_rate = df['rate'].mean()
            return f"Average utility rate across all ZIP codes: ${avg_rate:.4f} per kWh"
            
    except Exception as e:
        return f"Error retrieving utility rates: {str(e)}"

@tool
def get_energy_summary() -> str:
    """Get a summary of all available energy data."""
    try:
        summary_file = Path("energy_data_output/collection_summary.json")
        if not summary_file.exists():
            return "Energy data summary not found. Please run the data collection first."
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        result = "Energy Data Summary:\n"
        result += f"- Collected on: {summary['run_at']}\n"
        result += f"- EIA Retail Prices: {summary['sources']['eia_retail_prices']['rows']:,} rows\n"
        result += f"- EIA Generation: {summary['sources']['eia_generation']['rows']:,} rows\n"
        result += f"- OpenEI Utility Rates: {summary['sources']['openei_urdb']['rates']} rates\n"
        result += f"- NREL Utility Rates: {summary['sources']['nrel_utility_rates']['rows']} rows\n"
        result += f"- Collection duration: {summary['duration_seconds']:.1f} seconds"
        
        return result
        
    except Exception as e:
        return f"Error retrieving energy summary: {str(e)}"

def create_energy_agent():
    """Create an energy data analysis agent."""
    
    # Define the tools
    tools = [
        get_electricity_prices,
        get_energy_generation,
        get_utility_rates,
        get_energy_summary
    ]
    
    # Create the agent with tools
    agent = create_agent(
        model="ollama:llama3.1:8b",
        tools=tools,
        system_prompt="""You are a helpful energy data analyst assistant with access to historical energy data tools.

IMPORTANT: You have access to tools that contain ACTUAL energy data from files. When users ask questions:
1. IMMEDIATELY call the appropriate tools to get the data
2. The tools contain real historical data - not "real-time" but actual collected data
3. Use the tools first, then provide analysis based on the results
4. Do NOT say you can't access data - you CAN access it through the tools

Available tools:
- get_electricity_prices: Get electricity prices by state and sector (contains actual historical price data)
- get_energy_generation: Get energy generation data by source and state (contains actual generation data)  
- get_utility_rates: Get utility rates by ZIP code (contains actual utility rate data)
- get_energy_summary: Get a summary of all available energy data

Example: When asked "What is the electricity price in California?", call get_electricity_prices with state="CA".""",
        checkpointer=checkpointer
    )
    
    return agent

if __name__ == "__main__":
    # Create the agent
    agent = create_energy_agent()
    
    # Example queries
    queries = [
        "What is the electricity price in California?",
        "How much solar energy is generated in Texas?",
        "What are the utility rates for ZIP code 90210?",
        "Give me a summary of all available energy data"
    ]
    
    print("Energy Data Agent Examples:")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            }, config={
                "configurable": {
                    "thread_id": "example-thread"
                }
            })
            print(f"Answer: {result['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\nAsk about energy data: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }, config={
                "configurable": {
                    "thread_id": "interactive-thread"
                }
            })
            print(f"Answer: {result['messages'][-1].content}")
        except Exception as e:
            print(f"Error: {str(e)}")
