"""
TrueMeter Energy Data Collector
================================
Pulls free public energy data from:
  1. EIA API v2        - Electricity prices, consumption, generation by state
  2. OpenEI URDB API   - Utility rate structures for 3,700+ U.S. utilities
  3. NREL Utility Rates API - Average $/kWh by zip code + utility name

Setup:
  pip install requests pandas

API Keys (all free):
  - EIA:    https://www.eia.gov/opendata/register.php
  - OpenEI: https://openei.org/services/api/signup/
  - NREL:   https://developer.nrel.gov/signup/

Usage:
  python truemeter_energy_data_collector.py

Outputs:
  - eia_electricity_prices.csv        (state-level retail prices over time)
  - eia_electricity_consumption.csv   (state-level monthly consumption)
  - openei_utility_rates.json         (full utility rate structures)
  - nrel_utility_rates_by_zip.csv     (avg $/kWh lookup by zip + utility)
  - collection_summary.json           (metadata about this run)
"""

import requests
import pandas as pd
import json
import time
import logging
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — load API keys from env file
# ─────────────────────────────────────────────
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
        log.error("env file not found. Please create it with your API keys.")
        sys.exit(1)
    return keys

api_keys = load_api_keys()
EIA_API_KEY    = api_keys.get("EIA_API_KEY", "YOUR_EIA_API_KEY")
OPENEI_API_KEY = api_keys.get("OPENEI_API_KEY", "YOUR_OPENEI_API_KEY")
NREL_API_KEY   = api_keys.get("NREL_API_KEY", "YOUR_NREL_API_KEY")

OUTPUT_DIR = Path("./energy_data_output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_get(url, params=None, retries=3, delay=1.5):
    """GET with retry logic and rate-limit courtesy delay."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(delay)  # be a good API citizen
            return resp
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(delay * (attempt + 1))
    log.error(f"All retries failed for {url}")
    return None


# ─────────────────────────────────────────────
# 1. EIA API — Retail Electricity Prices by State
#    Endpoint: /v2/electricity/retail-sales
#    Coverage: Monthly, all 50 states, residential/commercial/industrial
#    Docs: https://www.eia.gov/opendata/documentation.php
# ─────────────────────────────────────────────

def fetch_eia_retail_prices(start="2022-01", end=None, sector="COM"):
    """
    Fetch monthly retail electricity prices by state.
    sector options: RES (residential), COM (commercial), IND (industrial)
    Returns a DataFrame.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m")

    log.info(f"Fetching EIA retail electricity prices ({sector}) {start} → {end}")

    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "monthly",
        "data[]": ["price", "sales", "customers"],
        "facets[sectorid][]": sector,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 5000,
        "offset": 0,
    }

    all_rows = []
    while True:
        resp = safe_get(url, params=params)
        if resp is None:
            break
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        total = int(data["response"].get("total", 0))
        log.info(f"  Retrieved {len(all_rows)}/{total} rows")
        if len(all_rows) >= total:
            break
        params["offset"] += 5000

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.rename(columns={
            "period": "month",
            "stateid": "state",
            "stateDescription": "state_name",
            "sectorid": "sector",
            "sectorName": "sector_name",
            "price": "price_cents_per_kwh",
            "sales": "sales_million_kwh",
            "customers": "customer_count",
        }, inplace=True)
        df["source"] = "EIA_retail_sales"

    return df


# ─────────────────────────────────────────────
# 2. EIA API — Monthly Electricity Generation by State & Fuel
#    Endpoint: /v2/electricity/electric-power-operational-data
#    Coverage: Monthly, state, generation source (solar, wind, gas, etc.)
# ─────────────────────────────────────────────

def fetch_eia_generation(start="2022-01", end=None):
    """Fetch monthly net generation by state and fuel type."""
    if end is None:
        end = datetime.now().strftime("%Y-%m")

    log.info(f"Fetching EIA electricity generation {start} → {end}")

    url = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "monthly",
        "data[]": ["generation", "total-consumption"],
        "facets[sectorid][]": "99",   # all sectors
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 5000,
        "offset": 0,
    }

    all_rows = []
    while True:
        resp = safe_get(url, params=params)
        if resp is None:
            break
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        total = int(data["response"].get("total", 0))
        log.info(f"  Retrieved {len(all_rows)}/{total} rows")
        if len(all_rows) >= total:
            break
        params["offset"] += 5000

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["source"] = "EIA_generation"

    return df


# ─────────────────────────────────────────────
# 3. OpenEI Utility Rate Database (URDB)
#    Endpoint: https://api.openei.org/utility_rates
#    Coverage: 3,700+ U.S. utilities, full rate structures
#    (TOU, demand charges, tiered rates, fixed fees, etc.)
#    Docs: https://openei.org/services/doc/rest/util_rates/
# ─────────────────────────────────────────────

def fetch_openei_utility_rates(limit=200, sector="Commercial"):
    """
    Fetch utility rate structures from OpenEI URDB.
    sector options: Residential, Commercial, Industrial, Lighting
    Returns a list of rate dicts.
    """
    log.info(f"Fetching OpenEI utility rates (sector={sector}, limit={limit})")

    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 5,
        "format": "json",
        "api_key": OPENEI_API_KEY,
        "sector": sector,
        "detail": "full",
        "limit": min(limit, 200),  # API max per page is 200
        "offset": 0,
    }

    all_rates = []
    while len(all_rates) < limit:
        resp = safe_get(url, params=params)
        if resp is None:
            break
        data = resp.json()
        rates = data.get("items", [])
        if not rates:
            break
        all_rates.extend(rates)
        log.info(f"  Retrieved {len(all_rates)} rates so far")
        if len(rates) < params["limit"]:
            break  # last page
        params["offset"] += params["limit"]

    return all_rates


def flatten_openei_rates(rates):
    """
    Flatten the nested OpenEI rate structure into a clean DataFrame
    for quick analysis. Keeps the raw JSON as a separate output.
    """
    flat = []
    for r in rates:
        try:
            flat.append({
                "label": r.get("label"),
                "utility_name": r.get("utility"),
                "rate_name": r.get("name"),
                "sector": r.get("sector"),
                "state": r.get("state"),
                "startdate": r.get("startdate"),
                "enddate": r.get("enddate"),
                "fixed_monthly_charge": r.get("fixedchargefirstmeter"),
                "fixed_charge_units": r.get("fixedchargeunits"),
                "currency": r.get("currency"),
                "has_demand_charges": bool(r.get("demandratestructure")),
                "has_tou": bool(r.get("energyratestructure") and
                               any("period" in str(tier) for tier in r.get("energyratestructure", []))),
                "source": "OpenEI_URDB",
                "openei_uri": r.get("uri"),
            })
        except Exception as e:
            log.warning(f"Could not flatten rate {r.get('label')}: {e}")
    return pd.DataFrame(flat)


# ─────────────────────────────────────────────
# 4. NREL Utility Rates API — Average $/kWh by Location
#    Endpoint: https://developer.nrel.gov/api/utility_rates/v3.json
#    Coverage: Zip code → utility name + avg residential/commercial/industrial $/kWh
#    Docs: https://developer.nrel.gov/docs/electricity/utility-rates-v3/
# ─────────────────────────────────────────────

SAMPLE_ZIPCODES = [
    "94105",  # San Francisco, CA
    "10001",  # New York, NY
    "60601",  # Chicago, IL
    "77001",  # Houston, TX
    "85001",  # Phoenix, AZ
    "98101",  # Seattle, WA
    "30301",  # Atlanta, GA
    "02101",  # Boston, MA
    "80201",  # Denver, CO
    "33101",  # Miami, FL
    "97201",  # Portland, OR
    "89101",  # Las Vegas, NV
    "55401",  # Minneapolis, MN
    "48201",  # Detroit, MI
    "28201",  # Charlotte, NC
]


def fetch_nrel_utility_rates(zipcodes=None):
    """
    Fetch average utility rates ($/kWh) + utility name by zip code.
    Note: NREL API now requires lat/lon instead of address.
    Returns a DataFrame with one row per zip code.
    """
    if zipcodes is None:
        zipcodes = SAMPLE_ZIPCODES

    log.info(f"Fetching NREL utility rates for {len(zipcodes)} zip codes")

    url = "https://developer.nrel.gov/api/utility_rates/v3.json"
    rows = []

    # Simple zip code to lat/lon mapping for major cities
    # In production, you'd use a proper geocoding service
    zipcode_coords = {
        "94105": ("37.7749", "-122.4194"),  # San Francisco, CA
        "10001": ("40.7484", "-73.9857"),   # New York, NY
        "60601": ("41.8781", "-87.6298"),   # Chicago, IL
        "77001": ("29.7604", "-95.3698"),   # Houston, TX
        "85001": ("33.4484", "-112.0740"),  # Phoenix, AZ
        "98101": ("47.6062", "-122.3321"),  # Seattle, WA
        "30301": ("33.7490", "-84.3880"),   # Atlanta, GA
        "02101": ("42.3601", "-71.0589"),   # Boston, MA
        "80201": ("39.7392", "-104.9903"),  # Denver, CO
        "33101": ("25.7617", "-80.1918"),   # Miami, FL
        "97201": ("45.5152", "-122.6784"),  # Portland, OR
        "89101": ("36.1699", "-115.1398"),  # Las Vegas, NV
        "55401": ("44.9778", "-93.2650"),   # Minneapolis, MN
        "48201": ("42.3314", "-83.0458"),   # Detroit, MI
        "28201": ("35.2271", "-80.8431"),   # Charlotte, NC
    }

    for zipcode in zipcodes:
        if zipcode not in zipcode_coords:
            log.warning(f"No coordinates for zip {zipcode}, skipping")
            continue
            
        lat, lon = zipcode_coords[zipcode]
        
        params = {
            "api_key": NREL_API_KEY,
            "lat": lat,
            "lon": lon,
        }
        resp = safe_get(url, params=params, delay=0.5)
        if resp is None:
            continue

        data = resp.json()
        outputs = data.get("outputs", {})
        if not outputs:
            log.warning(f"No data for zip {zipcode}")
            continue

        rows.append({
            "zip_code": zipcode,
            "utility_name": outputs.get("utility_name"),
            "utility_info": outputs.get("utility_info"),
            "residential_rate_usd_per_kwh": outputs.get("residential"),
            "commercial_rate_usd_per_kwh": outputs.get("commercial"),
            "industrial_rate_usd_per_kwh": outputs.get("industrial"),
            "source": "NREL_utility_rates_v3",
        })
        log.info(f"  ✓ {zipcode} → {outputs.get('utility_name')}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    start_time = datetime.now()
    summary = {
        "run_at": start_time.isoformat(),
        "sources": {},
    }

    print("\n" + "="*60)
    print("  TrueMeter Energy Data Collector")
    print("="*60 + "\n")

    # --- 1. EIA Retail Prices ---
    print("📡 [1/4] EIA — Retail Electricity Prices (Commercial)")
    eia_prices = fetch_eia_retail_prices(start="2022-01", sector="COM")
    if not eia_prices.empty:
        out = OUTPUT_DIR / "eia_electricity_prices_commercial.csv"
        eia_prices.to_csv(out, index=False)
        log.info(f"  Saved {len(eia_prices)} rows → {out}")
        summary["sources"]["eia_retail_prices"] = {"rows": len(eia_prices), "file": str(out)}

    # --- 2. EIA Generation ---
    print("\n📡 [2/4] EIA — Electricity Generation by State & Fuel")
    eia_gen = fetch_eia_generation(start="2022-01")
    if not eia_gen.empty:
        out = OUTPUT_DIR / "eia_electricity_generation.csv"
        eia_gen.to_csv(out, index=False)
        log.info(f"  Saved {len(eia_gen)} rows → {out}")
        summary["sources"]["eia_generation"] = {"rows": len(eia_gen), "file": str(out)}

    # --- 3. OpenEI Utility Rates ---
    print("\n📡 [3/4] OpenEI — Utility Rate Structures (Commercial)")
    openei_rates = fetch_openei_utility_rates(limit=500, sector="Commercial")
    if openei_rates:
        # Save raw JSON (preserves full nested rate structure)
        raw_out = OUTPUT_DIR / "openei_utility_rates_raw.json"
        with open(raw_out, "w") as f:
            json.dump(openei_rates, f, indent=2, default=str)

        # Save flattened CSV for quick analysis
        flat_df = flatten_openei_rates(openei_rates)
        flat_out = OUTPUT_DIR / "openei_utility_rates_flat.csv"
        flat_df.to_csv(flat_out, index=False)
        log.info(f"  Saved {len(openei_rates)} rates → {raw_out} + {flat_out}")
        summary["sources"]["openei_urdb"] = {
            "rates": len(openei_rates),
            "raw_json": str(raw_out),
            "flat_csv": str(flat_out),
        }

    # --- 4. NREL Rates by Zip ---
    print("\n📡 [4/4] NREL — Average Utility Rates by Zip Code")
    nrel_df = fetch_nrel_utility_rates(SAMPLE_ZIPCODES)
    if not nrel_df.empty:
        out = OUTPUT_DIR / "nrel_utility_rates_by_zip.csv"
        nrel_df.to_csv(out, index=False)
        log.info(f"  Saved {len(nrel_df)} rows → {out}")
        summary["sources"]["nrel_utility_rates"] = {"rows": len(nrel_df), "file": str(out)}

    # --- Summary ---
    summary["duration_seconds"] = round((datetime.now() - start_time).total_seconds(), 1)
    summary_out = OUTPUT_DIR / "collection_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"  ✅ Done in {summary['duration_seconds']}s")
    print(f"  Output directory: {OUTPUT_DIR.resolve()}")
    print("="*60)
    for source, meta in summary["sources"].items():
        print(f"  • {source}: {meta}")
    print()


if __name__ == "__main__":
    main()