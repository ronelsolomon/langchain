#!/usr/bin/env python3
"""
Test script to verify API keys for energy data services.
Tests EIA, OpenEI, and NREL API endpoints.
"""

import requests
import sys
from pathlib import Path

# Load API keys from env file
def load_api_keys():
    env_file = Path("env")
    if not env_file.exists():
        print("❌ env file not found")
        sys.exit(1)
    
    keys = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    keys[key.strip()] = value.strip().strip('"\'')
    
    return keys

def test_eia_api(api_key):
    """Test EIA API v2 endpoint"""
    print("\n🔍 Testing EIA API...")
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[]": ["price"],
        "facets[sectorid][]": "COM",
        "start": "2024-01",
        "end": "2024-01",
        "length": 5,
        "offset": 0,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "response" in data and "data" in data["response"]:
            rows = data["response"]["data"]
            print(f"✅ EIA API working - Retrieved {len(rows)} records")
            return True
        else:
            print("❌ EIA API - Unexpected response format")
            return False
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            print("❌ EIA API - Invalid API key")
        elif resp.status_code == 429:
            print("❌ EIA API - Rate limit exceeded")
        else:
            print(f"❌ EIA API - HTTP {resp.status_code}: {e}")
        return False
    except Exception as e:
        print(f"❌ EIA API - Error: {e}")
        return False

def test_openei_api(api_key):
    """Test OpenEI URDB API"""
    print("\n🔍 Testing OpenEI API...")
    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 5,
        "format": "json",
        "api_key": api_key,
        "sector": "Commercial",
        "detail": "full",
        "limit": 5,
        "offset": 0,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "items" in data:
            rates = data["items"]
            print(f"✅ OpenEI API working - Retrieved {len(rates)} rates")
            return True
        else:
            print("❌ OpenEI API - Unexpected response format")
            return False
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            print("❌ OpenEI API - Invalid API key")
        elif resp.status_code == 429:
            print("❌ OpenEI API - Rate limit exceeded")
        else:
            print(f"❌ OpenEI API - HTTP {resp.status_code}: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenEI API - Error: {e}")
        return False

def test_nrel_api(api_key):
    """Test NREL Utility Rates API"""
    print("\n🔍 Testing NREL API...")
    url = "https://developer.nrel.gov/api/utility_rates/v3.json"
    params = {
        "api_key": api_key,
        "lat": "37.7749",  # San Francisco latitude
        "lon": "-122.4194",  # San Francisco longitude
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "outputs" in data and data["outputs"]:
            outputs = data["outputs"]
            utility = outputs.get("utility_name", "Unknown")
            residential = outputs.get("residential", "N/A")
            print(f"✅ NREL API working - Found utility: {utility}, Residential rate: ${residential}/kWh")
            return True
        else:
            print("❌ NREL API - No data in response")
            print(f"Response: {data}")
            return False
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            print("❌ NREL API - Invalid API key")
        elif resp.status_code == 429:
            print("❌ NREL API - Rate limit exceeded")
        else:
            print(f"❌ NREL API - HTTP {resp.status_code}: {e}")
            print(f"Response body: {resp.text}")
        return False
    except Exception as e:
        print(f"❌ NREL API - Error: {e}")
        return False

def main():
    print("🚀 Testing API Keys for Energy Data Services")
    print("=" * 50)
    
    # Load API keys
    keys = load_api_keys()
    
    required_keys = ["EIA_API_KEY", "OPENEI_API_KEY", "NREL_API_KEY"]
    missing_keys = [key for key in required_keys if key not in keys]
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        return
    
    print(f"✅ Found {len(keys)} API keys")
    
    # Test each API
    results = {}
    
    if "EIA_API_KEY" in keys:
        results["EIA"] = test_eia_api(keys["EIA_API_KEY"])
    
    if "OPENEI_API_KEY" in keys:
        results["OpenEI"] = test_openei_api(keys["OPENEI_API_KEY"])
    
    if "NREL_API_KEY" in keys:
        results["NREL"] = test_nrel_api(keys["NREL_API_KEY"])
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    working_count = sum(results.values())
    total_count = len(results)
    
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {service}: {'Working' if status else 'Failed'}")
    
    print(f"\n🎯 Overall: {working_count}/{total_count} APIs working")
    
    if working_count == total_count:
        print("🎉 All API keys are valid and working!")
    else:
        print("⚠️  Some API keys need attention")

if __name__ == "__main__":
    main()
