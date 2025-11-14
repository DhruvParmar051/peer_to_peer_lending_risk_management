from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")

fred = Fred(api_key=API_KEY)

states = {
    'AL': 'ALUR', 'AK': 'AKUR', 'AZ': 'AZUR', 'AR': 'ARUR', 'CA': 'CAUR',
    'CO': 'COUR', 'CT': 'CTUR', 'DE': 'DEUR', 'FL': 'FLUR', 'GA': 'GAUR',
    'HI': 'HIUR', 'ID': 'IDUR', 'IL': 'ILUR', 'IN': 'INUR', 'IA': 'IAUR',
    'KS': 'KSUR', 'KY': 'KYUR', 'LA': 'LAUR', 'ME': 'MEUR', 'MD': 'MDUR',
    'MA': 'MAUR', 'MI': 'MIUR', 'MN': 'MNUR', 'MS': 'MSUR', 'MO': 'MOUR',
    'MT': 'MTUR', 'NE': 'NEUR', 'NV': 'NVUR', 'NH': 'NHUR', 'NJ': 'NJUR',
    'NM': 'NMUR', 'NY': 'NYUR', 'NC': 'NCUR', 'ND': 'NDUR', 'OH': 'OHUR',
    'OK': 'OKUR', 'OR': 'ORUR', 'PA': 'PAUR', 'RI': 'RIUR', 'SC': 'SCUR',
    'SD': 'SDUR', 'TN': 'TNUR', 'TX': 'TXUR', 'UT': 'UTUR', 'VT': 'VTUR',
    'VA': 'VAUR', 'WA': 'WAUR', 'WV': 'WVUR', 'WI': 'WIUR', 'WY': 'WYUR'
}

dfs = []
for st, series_id in states.items():
    data = fred.get_series(series_id)
    temp = pd.DataFrame({
        'DATE': data.index,
        'unemployment_rate': data.values,
        'addr_state': st
    })
    dfs.append(temp)

fred_df = pd.concat(dfs)
fred_df['quarter'] = pd.to_datetime(fred_df['DATE']).dt.to_period('Q')
fred_df.to_parquet(r"data/unemployment_rate_by_state.csv", compression='gzip',index=False)

print("Saved unemployment_rate_by_state.csv")