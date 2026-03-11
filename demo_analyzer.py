#!/usr/bin/env python3
"""
Demo script for Energy Data Analyzer
Shows how to use the LangChain-powered energy analysis tool
"""

from energy_analyzer import EnergyDataAnalyzer

def demo_analysis():
    """Demonstrate the energy analyzer capabilities"""
    print("🚀 Energy Data Analyzer Demo")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = EnergyDataAnalyzer()
    
    # Load and process data
    print("\n📊 Loading energy data...")
    analyzer.load_energy_data()
    
    print("\n🔍 Creating vector store...")
    analyzer.create_vector_store()
    
    # Sample queries
    sample_queries = [
        "Which states have the highest electricity prices?",
        "What are the average residential rates across all locations?",
        "Which utilities have time-of-use pricing?",
        "What is the price range for commercial electricity?",
        "How many different fuel types are in the generation data?"
    ]
    
    print("\n🎯 Running Sample Queries:")
    print("-" * 30)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. {query}")
        try:
            result = analyzer.query_data(query)
            print(f"   {result}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Generate insights
    print("\n💡 Generating Key Insights:")
    print("-" * 30)
    try:
        insights = analyzer.generate_insights()
        print(insights)
    except Exception as e:
        print(f"Error generating insights: {e}")
    
    # Compare specific locations
    print("\n📊 Comparing Rates (Sample Locations):")
    print("-" * 30)
    sample_zips = ["94105", "10001", "60601"]  # SF, NYC, Chicago
    try:
        comparison = analyzer.compare_rates(sample_zips)
        print(comparison)
    except Exception as e:
        print(f"Error comparing rates: {e}")

if __name__ == "__main__":
    demo_analysis()
