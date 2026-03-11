# Energy Data Collection & Analysis with LangChain

A comprehensive system for collecting and analyzing energy data from multiple APIs using LangChain for intelligent querying and insights.

## 🚀 Features

### Data Collection
- **EIA API v2** - Electricity prices, consumption, generation by state
- **OpenEI URDB API** - Utility rate structures for 3,700+ U.S. utilities  
- **NREL Utility Rates API** - Average $/kWh by zip code + utility name

### LangChain Analysis
- **Natural Language Queries** - Ask questions about energy data in plain English
- **Semantic Search** - Vector-based search across all energy datasets
- **Automated Insights** - AI-powered analysis and trend identification
- **Rate Comparisons** - Compare utility rates across different locations
- **Interactive Mode** - Command-line interface for real-time querying

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For local LLM support (recommended)
# Install Ollama: https://ollama.ai/
# Pull a model: ollama pull llama3:latest

# For OpenAI support (alternative)
# Add OPENAI_API_KEY to your env file
```

## 🔑 Setup API Keys

Create an `env` file with your API keys:

```
EIA_API_KEY    = "your_eia_api_key"
OPENEI_API_KEY = "your_openei_api_key"
NREL_API_KEY   = "your_nrel_api_key"
OPENAI_API_KEY = "your_openai_api_key"  # Optional, for OpenAI models
```

### Get API Keys (all free):
- **EIA**: https://www.eia.gov/opendata/register.php
- **OpenEI**: https://openei.org/services/api/signup/
- **NREL**: https://developer.nrel.gov/signup/

## 📊 Usage

### 1. Test API Keys
```bash
python test_api_keys.py
```

### 2. Collect Energy Data
```bash
python crawl.py
```
This creates CSV/JSON files in `energy_data_output/`:
- `eia_electricity_prices_commercial.csv`
- `eia_electricity_generation.csv`
- `openei_utility_rates_raw.json`
- `openei_utility_rates_flat.csv`
- `nrel_utility_rates_by_zip.csv`

### 3. Analyze with LangChain

#### Interactive Mode
```bash
python energy_analyzer.py
```

#### Demo Analysis
```bash
python demo_analyzer.py
```

#### Sample Queries
- "Which states have the highest electricity prices?"
- "What are the average residential rates?"
- "Compare rates in California vs Texas"
- "Which utilities have time-of-use pricing?"
- "What fuel types generate the most electricity?"

## 🧠 LangChain Architecture

The analyzer uses several LangChain components:

- **Document Loaders** - Convert CSV/JSON data to LangChain documents
- **Text Splitters** - Chunk data for optimal processing
- **Vector Stores** - FAISS for semantic similarity search
- **LLMs** - Ollama (local) or OpenAI (cloud) for analysis
- **Chains** - RAG (Retrieval-Augmented Generation) for accurate responses
- **Prompts** - Structured templates for consistent analysis

## 📁 Project Structure

```
langchain/
├── crawl.py                    # Data collection script
├── test_api_keys.py           # API key validation
├── energy_analyzer.py         # Main LangChain analyzer
├── demo_analyzer.py           # Demo and examples
├── env                        # API keys configuration
├── requirements.txt           # Python dependencies
├── energy_data_output/        # Collected data
│   ├── eia_electricity_prices_commercial.csv
│   ├── eia_electricity_generation.csv
│   ├── openei_utility_rates_flat.csv
│   └── nrel_utility_rates_by_zip.csv
└── README.md                  # This file
```

## 🔧 Configuration

### LLM Options

**Ollama (Recommended - Free & Local)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3:latest
```

**OpenAI (Alternative - Requires API Key)**
Add to `env` file:
```
OPENAI_API_KEY = "sk-..."
```

### Customization

- Modify `SAMPLE_ZIPCODES` in `crawl.py` to analyze different locations
- Adjust date ranges in data collection functions
- Extend prompts in `energy_analyzer.py` for specific analysis types

## 📈 Example Output

```
🚀 Energy Data Analyzer with LangChain
==================================================
📊 Loading energy data...
✅ Loaded 4 data summaries
🔍 Creating vector store for semantic search...
✅ Created vector store with 12 chunks

🎯 Interactive Query Mode
Type 'quit' to exit, 'insights' for general analysis

🔍 Ask about energy data: Which states have the highest electricity prices?

Based on the EIA commercial electricity prices data, the states with the highest 
average commercial electricity prices are:

1. Hawaii - Approximately 25-30 cents/kWh (highest due to island geography)
2. Alaska - Around 18-22 cents/kWh (remote location, higher distribution costs)
3. Connecticut - Roughly 16-18 cents/kWh (New England region)
4. Massachusetts - About 15-17 cents/kWh
5. Rhode Island - Similar to Massachusetts, 15-17 cents/kWh

These higher prices are typically due to factors like geographic isolation, 
higher infrastructure costs, and regional energy policies.
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

