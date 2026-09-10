"""
Monte Carlo Kostnadssimulator - OMSTRUKTURERT
Basert på Usikkerhetsanalyse-modellen med multiplikator-baserte usikkerhetsdrivere

Kjøring:
    streamlit run usikkerhetsanalyse_simulator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from datetime import datetime
from scipy.stats import beta
from scipy.optimize import fsolve
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Usikkerhetsanalyse Simulator", layout="wide")

st.title("Usikkerhetsanalyse Simulator")
st.markdown("**Multiplikator-baserte usikkerhetsdrivere** | Beta-PERT fordeling | Forventningsverdi-analyse")

# ============================================
# KONFIGURERING
# ============================================

STORAGE_FILE = "usikkerhet_data.json"
rng = np.random.RandomState(42)


SHEET_NAME = "Usikkerhetsanalyse Data"

@st.cache_resource
def get_gsheet_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return gspread.authorize(creds)

def load_data_from_sheet():
    try:
        sh = get_gsheet_client().open(SHEET_NAME)
    except Exception:
        return None

    try:
        pns_records = sh.worksheet("PNS").get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        pns_records = []
    try:
        u_records = sh.worksheet("Usikkerhetsdrivere").get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        u_records = []

    if not pns_records and not u_records:
        return None

    pns_list = [{
        "id": str(r.get("id", "")),
        "navn": r.get("navn", ""),
        "p10": float(r.get("p10", 0) or 0),
        "p50": float(r.get("p50", 0) or 0),
        "p90": float(r.get("p90", 0) or 0),
    } for r in pns_records]

    u_driver_list = []
    for r in u_records:
        try:
            andeler = json.loads(r.get("andeler_json", "") or "{}")
        except json.JSONDecodeError:
            andeler = {}
        u_driver_list.append({
            "id": str(r.get("id", "")),
            "navn": r.get("navn", ""),
            "p10": float(r.get("p10", 0) or 0),
            "p50": float(r.get("p50", 0) or 0),
            "p90": float(r.get("p90", 0) or 0),
            "andeler": andeler,
        })

    return {"pns_list": pns_list, "u_driver_list": u_driver_list}

def save_data_to_sheet(data):
    sh = get_gsheet_client().open(SHEET_NAME)

    try:
        pns_ws = sh.worksheet("PNS")
        pns_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        pns_ws = sh.add_worksheet(title="PNS", rows=200, cols=10)
    pns_rows = [["id", "navn", "p10", "p50", "p90"]] + [
        [p["id"], p["navn"], p["p10"], p["p50"], p["p90"]]
        for p in data.get("pns_list", [])
    ]
    pns_ws.update(pns_rows)

    try:
        u_ws = sh.worksheet("Usikkerhetsdrivere")
        u_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        u_ws = sh.add_worksheet(title="Usikkerhetsdrivere", rows=200, cols=10)
    u_rows = [["id", "navn", "p10", "p50", "p90", "andeler_json"]] + [
        [u["id"], u["navn"], u["p10"], u["p50"], u["p90"],
         json.dumps(u.get("andeler", {}), ensure_ascii=False)]
        for u in data.get("u_driver_list", [])
    ]
    u_ws.update(u_rows)


# ============================================
# FUNKSJONER - BETA-PERT
# ============================================

def fit_pert_alt(mode, p10_val, p90_val, lam=4):
    """Tilpasser Beta-PERT fordeling til P10/mode/P90"""
    def equations(ab):
        a, b = ab
        if b <= a or not (a <= mode <= b):
            return [1e6, 1e6]
        alpha = 1 + lam * (mode - a) / (b - a)
        beta_param = 1 + lam * (b - mode) / (b - a)
        cdf10 = beta.cdf((p10_val - a) / (b - a), alpha, beta_param)
        cdf90 = beta.cdf((p90_val - a) / (b - a), alpha, beta_param)
        return [cdf10 - 0.10, cdf90 - 0.90]

    span = p90_val - p10_val
    a0 = p10_val - 0.3 * span
    b0 = p90_val + 0.3 * span
    a, b = fsolve(equations, [a0, b0])
    return a, b

def riskpertalt_simuler(p10, mest_sannsynlig, p90, n=10000):
    """RiskPertAlt simulering"""
    if p10 == mest_sannsynlig == p90:
        return np.full(n, p10)
    
    a, b = fit_pert_alt(mest_sannsynlig, p10, p90)
    alpha = 1 + 4 * (mest_sannsynlig - a) / (b - a)
    beta_param = 1 + 4 * (b - mest_sannsynlig) / (b - a)
    
    return a + (b - a) * beta.rvs(alpha, beta_param, size=n)

def triangular_sample(p10, p50, p90, n=1):
    """Triangular distribution sampling for PNSer"""
    samples = []
    for _ in range(n):
        u = np.random.random()
        mode = p50
        a = p10
        b = p90
        f = (mode - a) / (b - a) if (b - a) > 0 else 0.5
        
        if u < f:
            sample = a + np.sqrt(u * (b - a) * (mode - a)) if (b - a) > 0 else a
        else:
            sample = b - np.sqrt((1 - u) * (b - a) * (b - mode)) if (b - a) > 0 else b
        
        samples.append(sample)
    
    return samples[0] if n == 1 else samples

def compute_u_expected_value(u_p10, u_p50, u_p90, basis, n_samples=10000):
    """
    Beregn forventningsverdi for usikkerhetsdriver ved hjelp av RiskPertAlt-simulering.
    
    Input:
    - u_p10, u_p50, u_p90: multiplikator-persentiler (f.eks. 0.95, 1.00, 1.05)
    - basis: beregningsgrunnlag (vektet sum av PNS-verdier basert på andeler)
    - n_samples: antall simuleringer
    
    Output:
    - forventet_verdi: gjennomsnitt av alle effekter
    - stddev: standardavvik av alle effekter
    - samples: alle simulerte effekter (for debug)
    """
    if u_p10 == u_p50 == u_p90:
        return 0.0, 0.0, np.zeros(n_samples)
    
    if basis == 0:  # Hvis beregningsgrunnlag er 0, ingen effekt
        return 0.0, 0.0, np.zeros(n_samples)
    
    # Simuler multiplikatorene med RiskPertAlt
    a, b = fit_pert_alt(u_p50, u_p10, u_p90)
    alpha = 1 + 4 * (u_p50 - a) / (b - a)
    beta_param = 1 + 4 * (b - u_p50) / (b - a)
    
    # Generer samples av multiplikatorer
    mult_samples = a + (b - a) * beta.rvs(alpha, beta_param, size=n_samples, random_state=rng)
    
    # Konverter til effekter: (multiplikator - 1) × basis
    effect_samples = (mult_samples - 1.0) * basis
    
    forventet_verdi = np.mean(effect_samples)
    stddev = np.std(effect_samples)
    
    return forventet_verdi, stddev, effect_samples

def load_data():
    """Last inn data fra JSON-fil"""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"pns_list": [], "u_driver_list": []}
    return {"pns_list": [], "u_driver_list": []}

def save_data(data):
    """Lagre data til JSON-fil"""
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    st.success("✅ Data lagret!")


def parse_import_data(text):
    """Parser CSV/Excel-format data"""
    lines = text.strip().split('\n')
    
    pns_data = []
    u_data = []
    
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 4:
            try:
                navn = parts[0]
                p10 = float(parts[1])
                p50 = float(parts[2])
                p90 = float(parts[3])
                
                if p10 > 0 or p50 > 0 or p90 > 0:
                    pns_data.append({
                        "navn": navn,
                        "p10": p10,
                        "p50": p50,
                        "p90": p90
                    })
            except:
                pass
    
    return pns_data, u_data

def format_currency(value, unit='kr'):
    """Formater valuta"""
    if unit == '1000kr':
        return f"{value/1000:,.0f} 1000 kr".replace(",", " ")
    elif unit == 'mill':
        return f"{value/1000000:,.2f} mill kr".replace(",", " ")
    else:
        return f"{value:,.0f} kr".replace(",", " ")

# ============================================
# INITIALISERING
# ============================================

if 'data' not in st.session_state:
    st.session_state.data = load_data()

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

if 'mva_prosent' not in st.session_state:
    st.session_state.mva_prosent = 0.0

if 'paloept' not in st.session_state:
    st.session_state.paloept = 0

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("⚙️ Innstillinger")
    
    num_simulations = st.slider(
        "Antall simuleringer",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    budget_ramme = st.number_input(
        "Kostnadsramme (kr)",
        value=415000000,
        step=1000000
    )
    
    currency_unit = st.selectbox(
        "Valutalav",
        ['kr', '1000kr', 'mill']
    )
    
    st.markdown("---")
    st.markdown(f"**Statistikk:**")
    st.write(f"📋 PNSer: {len(st.session_state.data['pns_list'])}")
    st.write(f"⚠️ Usikkerhetsdrivere: {len(st.session_state.data['u_driver_list'])}")
    
    st.markdown("---")
    
    if st.button("🚀 Kjør simulering", use_container_width=True):
        st.session_state.run_simulation = True
    
    if st.button("💾 Lagre data", use_container_width=True):
        save_data(st.session_state.data)

# ============================================
# MAIN TABS
# ============================================

tab0, tab1, tab2, tab3, tab3b, tab4, tab5 = st.tabs([
    "📥 Import Data",              # ← LEGG TIL DENNE
    "📋 PNSer",
    "⚠️ Usikkerhetsdrivere",
    "📋 Oppsummering",
    "⚙️ Oppsett",
    "📊 Resultater",
    "📈 Analyse"
])



# ============================================
# TAB: IMPORT DATA (Ny!)
# ============================================
 
import io
 
def parse_import_data(text):
    """Parser CSV/Excel-format data"""
    lines = text.strip().split('\n')
    
    pns_data = []
    u_data = []
    
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) >= 4:
            # Format: Navn, P10, P50, P90
            try:
                navn = parts[0]
                p10 = float(parts[1])
                p50 = float(parts[2])
                p90 = float(parts[3])
                
                # Hvis det er tall, det er PNS
                if p10 > 0 or p50 > 0 or p90 > 0:
                    pns_data.append({
                        "navn": navn,
                        "p10": p10,
                        "p50": p50,
                        "p90": p90
                    })
            except:
                pass
        
        elif len(parts) >= 4 and 'U' in parts[0].upper():
            # Format: U-driver, P10_mult, P50_mult, P90_mult
            # Eksempel: "U2 Lokale forhold, 0.95, 1.00, 1.05"
            try:
                navn = parts[0]
                p10 = float(parts[1])
                p50 = float(parts[2])
                p90 = float(parts[3])
                
                u_data.append({
                    "navn": navn,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90
                })
            except:
                pass
    
    return pns_data, u_data
 
 
# LEGG DENNE TABEN I TABS-LISTEN:
# tab1, tab2, tab3, tab3b, tab4, tab5 = st.tabs([...])
# ENDRE TIL:
# tab0, tab1, tab2, tab3, tab3b, tab4, tab5 = st.tabs([
#     "📥 Import Data",  ← LEGG TIL
#     "📋 PNSer",
#     ... rest
 
# Deretter, LEGG TIL DENNE KODEN ETTER tabs-listen:
 
with tab0:
    st.header("📥 Importer data fra Excel")
    st.info("""
    **Slik bruker du:**
    1. Kopier data fra Excel (eller skriv inn under)
    2. Format: 
       - **PNSer:** Navn, P10, P50, P90
       - **Usikkerhetsdrivere:** Navn, P10_mult, P50_mult, P90_mult
    3. En rad per element
    4. Klikk "Importer"
    
    **Eksempel - PNSer:**
    ```
    Grunnarbeid, 5000000, 6000000, 7000000
    Mudringsarbeider, 0, 0, 500000
    Forurensede masser, 2000000, 3000000, 4000000
    ```
    
    **Eksempel - Usikkerhetsdrivere:**
    ```
    U2 Lokale forhold, 0.95, 1.00, 1.05
    U3 Interessenter, 0.98, 1.00, 1.02
    ```
    """)
    
    st.markdown("---")
    
    # Velg type
    import_type = st.radio("Hva vil du importere?", ["PNSer", "Usikkerhetsdrivere"])
    
    st.markdown("---")
    
    # Text area for paste
    st.subheader("Lim inn data her")
    import_text = st.text_area(
        "Kopier fra Excel (navn, P10, P50, P90), en per linje",
        height=200,
        placeholder="Eksempel:\nGrunnarbeid, 5000000, 6000000, 7000000\nMudringsarbeider, 0, 0, 500000"
    )
    
    # Import knapp
    if st.button("🚀 Importer data", use_container_width=True):
        if import_text.strip():
            pns_list, u_list = parse_import_data(import_text)
            
            if import_type == "PNSer" and pns_list:
                for pns in pns_list:
                    # Sjekk om den allerede finnes
                    exists = any(p['navn'] == pns['navn'] for p in st.session_state.data['pns_list'])
                    
                    if not exists:
                        pns['id'] = int(datetime.now().timestamp())
                        st.session_state.data['pns_list'].append(pns)
                        st.success(f"✅ {pns['navn']} lagt til!")
                    else:
                        st.warning(f"⚠️ {pns['navn']} finnes allerede")
                
                save_data(st.session_state.data)
                st.success(f"✅ {len(pns_list)} PNSer importert!")
                st.rerun()
            
            elif import_type == "Usikkerhetsdrivere" and u_list:
                for u in u_list:
                    # Sjekk om den allerede finnes
                    exists = any(d['navn'] == u['navn'] for d in st.session_state.data['u_driver_list'])
                    
                    if not exists:
                        u['id'] = int(datetime.now().timestamp())
                        u['andeler'] = {}  # Tom dict for andeler
                        st.session_state.data['u_driver_list'].append(u)
                        st.success(f"✅ {u['navn']} lagt til!")
                    else:
                        st.warning(f"⚠️ {u['navn']} finnes allerede")
                
                save_data(st.session_state.data)
                st.success(f"✅ {len(u_list)} usikkerhetsdrivere importert!")
                st.rerun()
            
            else:
                st.error("❌ Ingen data funnet. Sjekk formatet!")
        else:
            st.error("❌ Lim inn data først!")
    
    st.markdown("---")
    
    st.subheader("📋 Quick template - kopier dette til Excel")
    
    if import_type == "PNSer":
        template = """Navn,P10,P50,P90
Kostnad 1,1000000,2000000,3000000
Kostnad 2,500000,1000000,1500000
Kostnad 3,2000000,3000000,5000000"""
    else:
        template = """Navn,P10,P50,P90
U1 Driver,0.95,1.00,1.10
U2 Driver,0.90,1.00,1.15
U3 Driver,0.98,1.00,1.05"""
    
    st.code(template, language="text")
    
    # Copy-knapp
    st.write("**Kopier template over til Excel, fyll inn, kopier tilbake og lim inn i boksen over!**")
 

# ============================================
# TAB 1: PNSer
# ============================================

with tab1:
    st.header("📋 PNSer")
    st.info("Legg til PNSer med triple estimater (P10, P50, P90)")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Legg til PNS", use_container_width=True):
            new_pns = {
                "id": int(datetime.now().timestamp()),
                "navn": f"PNS {len(st.session_state.data['pns_list']) + 1}",
                "p10": 0,
                "p50": 1000000,
                "p90": 2000000
            }
            st.session_state.data['pns_list'].append(new_pns)
            save_data(st.session_state.data)
            st.rerun()
    
    if len(st.session_state.data['pns_list']) > 0:
        for i, pns in enumerate(st.session_state.data['pns_list']):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])
            
            with col1:
                pns['navn'] = st.text_input(
                    "Navn",
                    value=pns['navn'],
                    key=f"pns_navn_{pns['id']}",
                    label_visibility="collapsed"
                )
            with col2:
                pns['p10'] = st.number_input(
                    "P10",
                    value=int(pns['p10']),
                    step=100000,
                    key=f"pns_p10_{pns['id']}",
                    label_visibility="collapsed"
                )
            with col3:
                pns['p50'] = st.number_input(
                    "P50",
                    value=int(pns['p50']),
                    step=100000,
                    key=f"pns_p50_{pns['id']}",
                    label_visibility="collapsed"
                )
            with col4:
                pns['p90'] = st.number_input(
                    "P90",
                    value=int(pns['p90']),
                    step=100000,
                    key=f"pns_p90_{pns['id']}",
                    label_visibility="collapsed"
                )
            with col5:
                if st.button("🗑️", key=f"del_pns_{pns['id']}"):
                    st.session_state.data['pns_list'].remove(pns)
                    save_data(st.session_state.data)
                    st.rerun()
        
        # Tabell
        st.markdown("---")
        pns_df = pd.DataFrame([
            {
                "Navn": p['navn'],
                "P10": f"{p['p10']:,}".replace(",", " "),
                "P50": f"{p['p50']:,}".replace(",", " "),
                "P90": f"{p['p90']:,}".replace(",", " "),
            }
            for p in st.session_state.data['pns_list']
        ])
        st.dataframe(pns_df, use_container_width=True, hide_index=True)
        
        total_p50 = sum(p['p50'] for p in st.session_state.data['pns_list'])
        st.metric("Total P50 (PNSer)", format_currency(total_p50, currency_unit))
    else:
        st.warning("📭 Ingen PNSer. Klikk '➕ Legg til PNS'")

# ============================================
# TAB 2: USIKKERHETSDRIVERE (MULTIPLIKATOR)
# ============================================

with tab2:
    st.header("⚠️ Usikkerhetsdrivere (Multiplikator-basert)")
    st.info("Usikkerhetsdriverne er multiplikatorer (f.eks. 1.00 = 100%, 1.041 = 104.1%). De påvirker andeler av hver PNS.")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Legg til driver", use_container_width=True):
            new_u = {
                "id": int(datetime.now().timestamp()),
                "navn": f"Usikkerhetsdriver {len(st.session_state.data['u_driver_list']) + 1}",
                "p10": 1.00,
                "p50": 1.00,
                "p90": 1.041,
                "andeler": {}  # Dictionary: pns_id -> andel (0-1)
            }
            st.session_state.data['u_driver_list'].append(new_u)
            save_data(st.session_state.data)
            st.rerun()
    
    if len(st.session_state.data['u_driver_list']) > 0:
        for u in st.session_state.data['u_driver_list']:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])
            
            with col1:
                u['navn'] = st.text_input(
                    "Navn",
                    value=u['navn'],
                    key=f"u_navn_{u['id']}",
                    label_visibility="collapsed"
                )
            with col2:
                u['p10'] = st.number_input(
                    "P10 (multiplikator)",
                    value=float(u['p10']),
                    step=0.01,
                    key=f"u_p10_{u['id']}",
                    label_visibility="collapsed"
                )
            with col3:
                u['p50'] = st.number_input(
                    "P50 (multiplikator)",
                    value=float(u['p50']),
                    step=0.01,
                    key=f"u_p50_{u['id']}",
                    label_visibility="collapsed"
                )
            with col4:
                u['p90'] = st.number_input(
                    "P90 (multiplikator)",
                    value=float(u['p90']),
                    step=0.01,
                    key=f"u_p90_{u['id']}",
                    label_visibility="collapsed"
                )
            with col5:
                if st.button("🗑️", key=f"del_u_{u['id']}"):
                    st.session_state.data['u_driver_list'].remove(u)
                    save_data(st.session_state.data)
                    st.rerun()
        
        # ===== ANDELER PER PNS =====
        st.markdown("---")
        st.subheader("📊 Andel av hver PNS påvirket av driver")
        
        for u in st.session_state.data['u_driver_list']:
            st.markdown(f"**{u['navn']}**")
            
            # Initialisér andeler hvis de ikke eksisterer
            if 'andeler' not in u:
                u['andeler'] = {}
            
            andel_cols = st.columns(len(st.session_state.data['pns_list']))
            
            for idx, pns in enumerate(st.session_state.data['pns_list']):
                pns_id_str = str(pns['id'])
                current_andel = u['andeler'].get(pns_id_str, 1.0)  # Default 100%
                
                with andel_cols[idx]:
                    def save_andel(u_id=u['id'], pns_id=pns['id']):
                        save_data(st.session_state.data)
                    
                    new_andel = st.slider(
                        f"{pns['navn']}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_andel),
                        step=0.05,
                        key=f"andel_{u['id']}_{pns['id']}",
                        label_visibility="collapsed",
                        on_change=save_andel
                    )
                    u['andeler'][pns_id_str] = new_andel
                    
                    # Beregn og vis beregningsgrunnlag
                    beregningsgrunnlag = pns['p50'] * new_andel
                    st.markdown(f"<p style='text-align: center; font-weight: bold;'>{beregningsgrunnlag:,.0f}</p>".replace(",", " "), unsafe_allow_html=True)
            
            st.write("")  # spacing
        
        # Lagre etter alle andeler er oppdatert
        save_data(st.session_state.data)
        u_df = pd.DataFrame([
            {
                "Navn": u['navn'],
                "P10": f"{u['p10']:.3f}",
                "P50": f"{u['p50']:.3f}",
                "P90": f"{u['p90']:.3f}",
            }
            for u in st.session_state.data['u_driver_list']
        ])
        st.dataframe(u_df, use_container_width=True, hide_index=True)
    else:
        st.warning("📭 Ingen usikkerhetsdrivere. Klikk '➕ Legg til driver'")

# ============================================
# TAB 2B: OPPSUMMERING - FØR SIMULERING
# ============================================

with tab2b:
    st.header("📋 Oppsummering - Før Simulering")
    st.info("Se hele kostnadsbildet før du kjører simulering")
    
    # MVA og Påløpt
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.mva_prosent = st.number_input(
            "MVA (%)",
            value=float(st.session_state.mva_prosent),
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            help="Mva-sats (f.eks. 25% for standard mva)"
        )
    
    with col2:
        st.session_state.paloept = st.number_input(
            "Påløpt (kr)",
            value=int(st.session_state.paloept),
            step=1000000,
            help="Påløpte kostnader som skal trekkes fra"
        )
    
    st.markdown("---")
    
    # ===== TABELL OVER PNSer =====
    st.subheader("📋 PNSer - Triple Estimater")
    
    pns_total_p50 = sum(p['p50'] for p in st.session_state.data['pns_list'])
    
    pns_display_data = []
    for pns in st.session_state.data['pns_list']:
        pns_display_data.append({
            'Navn': pns['navn'],
            'P10': f"{pns['p10']:,}".replace(",", " "),
            'P50': f"{pns['p50']:,}".replace(",", " "),
            'P90': f"{pns['p90']:,}".replace(",", " ")
        })
    
    pns_df = pd.DataFrame(pns_display_data)
    st.dataframe(pns_df, use_container_width=True, hide_index=True)
    
    st.metric("Sum P50 (PNSer)", format_currency(pns_total_p50, currency_unit))
    
    st.markdown("---")
    
    # ===== TABELL OVER USIKKERHETSDRIVERE =====
    st.subheader("⚠️ Usikkerhetsdrivere - Med Forventingsverdi")
    
    u_display_data = []
    total_u_expected = 0
    
    for u in st.session_state.data['u_driver_list']:
        # Beregn beregningsgrunnlag basert på andeler
        beregningsgrunnlag = 0
        for pns in st.session_state.data['pns_list']:
            pns_id_str = str(pns['id'])
            andel = u.get('andeler', {}).get(pns_id_str, 1.0)  # Default 100% hvis ikke satt
            beregningsgrunnlag += pns['p50'] * andel
        
        # Beregn P10/P50/P90 som kroneeffekter
        p10_effekt = beregningsgrunnlag * (u['p10'] - 1)
        p50_effekt = beregningsgrunnlag * (u['p50'] - 1)
        p90_effekt = beregningsgrunnlag * (u['p90'] - 1)
        
        # Beregn forventningsverdi ved RiskPertAlt-simulering
        forventet_effekt, stddev, _ = compute_u_expected_value(
            u['p10'], u['p50'], u['p90'], 
            beregningsgrunnlag, 
            n_samples=10000
        )
        
        total_u_expected += forventet_effekt
        
        u_display_data.append({
            'Navn': u['navn'],
            'P10 (%)': f"{(u['p10']-1)*100:.1f}%",
            'P50 (%)': f"{(u['p50']-1)*100:.1f}%",
            'P90 (%)': f"{(u['p90']-1)*100:.1f}%",
            'P10 (kr)': f"{p10_effekt:,.0f}".replace(",", " "),
            'P50 (kr)': f"{p50_effekt:,.0f}".replace(",", " "),
            'P90 (kr)': f"{p90_effekt:,.0f}".replace(",", " "),
            'Forventet (kr)': f"{forventet_effekt:,.0f}".replace(",", " "),
        })
    
    u_df = pd.DataFrame(u_display_data)
    st.dataframe(u_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ===== TOTALER =====
    st.subheader("💰 Totalkostnad")
    
    subtotal = pns_total_p50 + total_u_expected
    mva_belop = subtotal * (st.session_state.mva_prosent / 100)
    total_med_mva = subtotal + mva_belop
    total_etter_paloept = total_med_mva - st.session_state.paloept
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PNS Sum (P50)", format_currency(pns_total_p50, currency_unit))
    with col2:
        st.metric("Usikkerhets-effekt", format_currency(total_u_expected, currency_unit))
    with col3:
        st.metric(f"Med MVA ({st.session_state.mva_prosent:.1f}%)", format_currency(total_med_mva, currency_unit))
    with col4:
        st.metric("Etter påløpt", format_currency(total_etter_paloept, currency_unit))
    
    st.markdown("---")
    
    # Detaljert tabell
    st.subheader("📈 Kostnadsoversikt")
    
    oversikt_data = {
        'Beskrivelse': [
            'PNS Basisestimat (P50)',
            'Usikkerhetseffekt (forventet)',
            'Subtotal',
            f'MVA ({st.session_state.mva_prosent:.1f}%)',
            'Total med MVA',
            'Påløpt',
            'Netto totalkostnad'
        ],
        'Beløp (kr)': [
            pns_total_p50,
            total_u_expected,
            subtotal,
            mva_belop,
            total_med_mva,
            st.session_state.paloept,
            total_etter_paloept
        ]
    }
    
    oversikt_df = pd.DataFrame(oversikt_data)
    oversikt_df['Beløp (1000 kr)'] = (oversikt_df['Beløp (kr)'] / 1000).round(0).astype(int)
    oversikt_df_display = oversikt_df[['Beskrivelse', 'Beløp (1000 kr)']].copy()
    oversikt_df_display['Beløp (1000 kr)'] = oversikt_df_display['Beløp (1000 kr)'].apply(lambda x: f"{x:,}".replace(",", " "))
    
    st.dataframe(oversikt_df_display, use_container_width=True, hide_index=True)
    
    st.info(f"✅ Klar for simulering? Gå til **⚙️ Oppsett**-fanen og klikk **🚀 KJØR SIMULERING**")

# ============================================
# TAB 3: OPPSETT & SIMULERING
# ============================================

with tab3:
    st.header("⚙️ Simuleringoppsett")
    
    st.info(f"**Antall simuleringer:** {num_simulations:,} | **Kostnadsramme:** {format_currency(budget_ramme, currency_unit)}".replace(",", " "))
    
    if st.button("🚀 KJØR SIMULERING", use_container_width=True, type="primary"):
        if len(st.session_state.data['pns_list']) == 0:
            st.error("⚠️ Legg til minst en PNS!")
        else:
            with st.spinner(f"Kjører {num_simulations:,} simuleringer...".replace(",", " ")):
                results = []
                
                for sim in range(num_simulations):
                    total = 0
                    
                    # PNSer - simuler hver enkelt
                    pns_samples = {}
                    for pns in st.session_state.data['pns_list']:
                        pns_value = triangular_sample(pns['p10'], pns['p50'], pns['p90'])
                        pns_samples[str(pns['id'])] = pns_value
                        total += pns_value
                    
                    # Usikkerhetsdrivere - simuler hver med andeler
                    for u in st.session_state.data['u_driver_list']:
                        # Beregn beregningsgrunnlag basert på andeler
                        beregningsgrunnlag = 0
                        for pns in st.session_state.data['pns_list']:
                            pns_id_str = str(pns['id'])
                            pns_value = pns_samples.get(pns_id_str, 0)
                            andel = u.get('andeler', {}).get(pns_id_str, 1.0)
                            beregningsgrunnlag += pns_value * andel
                        
                        # Simuler multiplikator med RiskPertAlt
                        mult_sample = riskpertalt_simuler(u['p10'], u['p50'], u['p90'], n=1)[0]
                        u_effekt = (mult_sample - 1.0) * beregningsgrunnlag
                        total += u_effekt
                    
                    results.append(total)
                
                st.session_state.simulation_results = {
                    'samples': results,
                    'num_sims': num_simulations,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success("✅ Simulering ferdig!")
                st.balloons()

# ============================================
# TAB 4: RESULTATER
# ============================================

with tab4:
    st.header("📊 Simuleringsresultater")
    
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results['samples']
        sorted_results = sorted(results)
        
        p10 = sorted_results[int(len(sorted_results) * 0.1)]
        p50 = sorted_results[int(len(sorted_results) * 0.5)]
        p85 = sorted_results[int(len(sorted_results) * 0.85)]
        p90 = sorted_results[int(len(sorted_results) * 0.9)]
        mean = np.mean(sorted_results)
        stddev = np.std(sorted_results)
        
        over_prob = sum(1 for x in sorted_results if x > budget_ramme) / len(sorted_results) * 100
        
        # Metric cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("P10", format_currency(p10, currency_unit))
        with col2:
            st.metric("P50", format_currency(p50, currency_unit))
        with col3:
            st.metric("P85", format_currency(p85, currency_unit))
        with col4:
            st.metric("Gjennomsnitt", format_currency(mean, currency_unit))
        with col5:
            st.metric("Std.avvik", format_currency(stddev, currency_unit))
        with col6:
            st.metric("Over ramme", f"{over_prob:.1f}%")
        
        # Detaljert tabell
        st.subheader("📊 Detaljerte Resultater")
        
        basisestimat = sum(p['p50'] for p in st.session_state.data['pns_list'])
        forventet_tillegg = mean - basisestimat
        usikkerhetsavsetning = p85 - p50
        
        resultat_data = {
            'Beskrivelse': [
                'Basisestimat',
                'Forventet tillegg (Gjennomsnitt - Basisestimat)',
                'P50 (Median)',
                'Usikkerhetsavsetning (P85-P50)',
                'P85',
                'Forventningsverdi (Gjennomsnitt)',
                'Standardavvik'
            ],
            'Totalt': [
                basisestimat,
                forventet_tillegg,
                p50,
                usikkerhetsavsetning,
                p85,
                mean,
                stddev
            ],
            '%': [
                '-',
                f"{(forventet_tillegg / basisestimat * 100):.2f}%" if basisestimat > 0 else '-',
                '-',
                f"{(usikkerhetsavsetning / p50 * 100):.2f}%" if p50 > 0 else '-',
                '-',
                '-',
                f"{(stddev / mean * 100):.2f}%" if mean > 0 else '-'
            ]
        }
        
        resultat_df = pd.DataFrame(resultat_data)
        
        resultat_display = []
        for idx, row in resultat_df.iterrows():
            resultat_display.append({
                'Beskrivelse': row['Beskrivelse'],
                'Totalt (1000 kr)': f"{row['Totalt'] / 1000:,.0f}".replace(",", " "),
                '%': row['%']
            })
        
        resultat_display_df = pd.DataFrame(resultat_display)
        st.dataframe(resultat_display_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Histogram
        st.subheader("📊 Kostnadsfordeling (Histogram)")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        divisor = 1 if currency_unit == 'kr' else (1000 if currency_unit == '1000kr' else 1000000)
        ax.hist([x / divisor for x in sorted_results], bins=50, color='#4278f5', alpha=0.7, edgecolor='black')
        ax.axvline(p10 / divisor, color='green', linestyle='--', linewidth=2, label='P10')
        ax.axvline(p50 / divisor, color='blue', linestyle='--', linewidth=2, label='P50')
        ax.axvline(p90 / divisor, color='orange', linestyle='--', linewidth=2, label='P90')
        ax.axvline(mean / divisor, color='red', linestyle='--', linewidth=2, label='Forventningsverdi')
        ax.set_xlabel(f'Verdi ({currency_unit})')
        ax.set_ylabel('Frekvens')
        ax.set_title('Kostnadsfordeling')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        # S-kurve
        st.subheader("📈 S-kurve (Kumulativ fordeling)")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results) * 100
        
        ax.plot([x / divisor for x in sorted_results], cumulative, color='#4a90d9', linewidth=2.5)
        ax.fill_between([x / divisor for x in sorted_results], cumulative - 1, cumulative + 1, 
                        color='#4a90d9', alpha=0.15)
        
        # Marker Basisestimat, P50, P85, P90
        # Beregn basisestimat
        basisestimat = sum(p['p50'] for p in st.session_state.data['pns_list'])
        
        ax.plot(basisestimat / divisor, 10, 'o', color='darkred', markersize=8)
        ax.annotate(f'Basisestimat\n{format_currency(basisestimat, currency_unit)}', 
                   xy=(basisestimat / divisor, 10),
                   xytext=(basisestimat / divisor - (budget_ramme / divisor * 0.1), 20),
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'),
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.plot(p50 / divisor, 50, 'o', color='darkred', markersize=8)
        ax.annotate(f'P50\n{format_currency(p50, currency_unit)}', 
                   xy=(p50 / divisor, 50),
                   xytext=(p50 / divisor + (budget_ramme / divisor * 0.05), 35),
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'),
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.plot(p85 / divisor, 85, 'o', color='darkred', markersize=8)
        ax.annotate(f'P85\n{format_currency(p85, currency_unit)}', 
                   xy=(p85 / divisor, 85),
                   xytext=(p85 / divisor + (budget_ramme / divisor * 0.05), 70),
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'),
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.axvline(budget_ramme / divisor, color='orange', linestyle='--', linewidth=2, label='Budsjettramme')
        ax.fill_between([x / divisor for x in sorted_results], cumulative, 
                        where=np.array([x / divisor for x in sorted_results]) >= budget_ramme / divisor,
                        alpha=0.2, color='red')
        
        ax.set_xlabel(f'Verdi ({currency_unit})', fontsize=11)
        ax.set_ylabel('Kumulativ sannsynlighet (%)', fontsize=11)
        ax.set_title(f'S-kurve | P(over ramme) = {over_prob:.1f}%', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticklabels([f'{y}%' for y in range(0, 101, 10)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        st.pyplot(fig, use_container_width=True)
        
        st.info(f"**Sannsynlighet for å overskride {format_currency(budget_ramme, currency_unit)}:** {over_prob:.2f}%")
    
    else:
        st.warning("📭 Kjør simulering i '⚙️ Oppsett'-fanen først")

# ============================================
# TAB 5: ANALYSE - TORNADO
# ============================================

with tab5:
    st.header("📊 Tornadodiagram - Sensitivitetsanalyse")
    
    if st.session_state.simulation_results:
        # Beregn tornado-data
        tornado_data = []
        
        # PNSer
        for pns in st.session_state.data['pns_list']:
            forventet_pns = (pns['p10'] + pns['p50'] + pns['p90']) / 3
            tornado_data.append({
                'Navn': pns['navn'],
                'Type': 'PNS',
                'P10': pns['p10'] - pns['p50'],
                'P50': 0,
                'P90': pns['p90'] - pns['p50'],
                'Forventet verdi': forventet_pns - pns['p50'],
                'Std.avvik': 0  # placeholder
            })
        
        # Usikkerhetsdrivere
        total_pns = sum(p['p50'] for p in st.session_state.data['pns_list'])
        for u in st.session_state.data['u_driver_list']:
            # Beregn beregningsgrunnlag basert på andeler
            beregningsgrunnlag = 0
            for pns in st.session_state.data['pns_list']:
                pns_id_str = str(pns['id'])
                andel = u.get('andeler', {}).get(pns_id_str, 1.0)
                beregningsgrunnlag += pns['p50'] * andel
            
            p10_effekt = beregningsgrunnlag * (u['p10'] - 1)
            p50_effekt = beregningsgrunnlag * (u['p50'] - 1)
            p90_effekt = beregningsgrunnlag * (u['p90'] - 1)
            
            # Beregn forventningsverdi ved RiskPertAlt-simulering
            forventet_effekt, stddev, _ = compute_u_expected_value(
                u['p10'], u['p50'], u['p90'], 
                beregningsgrunnlag, 
                n_samples=10000
            )
            
            tornado_data.append({
                'Navn': u['navn'],
                'Type': 'Usikkerhetsdriver',
                'P10': p10_effekt,
                'P50': p50_effekt,
                'P90': p90_effekt,
                'Forventet verdi': forventet_effekt,
                'Std.avvik': stddev
            })
        
        df_tornado = pd.DataFrame(tornado_data)
        df_tornado['Spennvidde'] = df_tornado['P90'] - df_tornado['P10']
        df_tornado_sorted = df_tornado.sort_values('Spennvidde', ascending=True)
        
        # Tornado diagram
        fig, ax = plt.subplots(figsize=(14, 10))
        
        divisor = 1 if currency_unit == 'kr' else (1000 if currency_unit == '1000kr' else 1000000)
        
        for i, (idx, row) in enumerate(df_tornado_sorted.iterrows()):
            p10_avvik = row['P10'] / divisor
            p90_avvik = row['P90'] / divisor
            
            if p10_avvik < 0:
                ax.barh(i, -p10_avvik, left=p10_avvik, color='#4a90d9', edgecolor='black', height=0.6, linewidth=1.5)
                ax.text(p10_avvik - (abs(p10_avvik) * 0.05), i, f'{p10_avvik:,.0f}', 
                       va='center', ha='right', fontsize=8, color='white', fontweight='bold')
            
            if p90_avvik > 0:
                ax.barh(i, p90_avvik, left=0, color='#8b1a1a', edgecolor='black', height=0.6, linewidth=1.5)
                ax.text(p90_avvik + (abs(p90_avvik) * 0.05), i, f'{p90_avvik:,.0f}', 
                       va='center', ha='left', fontsize=8, color='white', fontweight='bold')
        
        min_p10 = df_tornado_sorted['P10'].min() / divisor
        max_p90 = df_tornado_sorted['P90'].max() / divisor
        x_padding = 1_000_000 / divisor
        
        ax.axvline(0, color='black', linewidth=2)
        ax.set_yticks(range(len(df_tornado_sorted)))
        ax.set_yticklabels(df_tornado_sorted['Navn'], fontsize=10)
        ax.set_title('Tornadodiagram - Påvirkning på totalkostnad', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Avvik fra baseline ({currency_unit})', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlim(min_p10 - x_padding, max_p90 + x_padding)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Totalsummer
        col1, col2, col3 = st.columns(3)
        total_p10 = df_tornado_sorted['P10'].sum()
        total_p90 = df_tornado_sorted['P90'].sum()
        
        with col1:
            st.metric("🔵 Total Best Case (P10)", format_currency(total_p10, currency_unit))
        with col2:
            st.metric("🔴 Total Worst Case (P90)", format_currency(total_p90, currency_unit))
        with col3:
            st.metric("📊 Totalt spennvidde", format_currency(total_p90 - total_p10, currency_unit))
        
        # Tornado data tabell
        st.subheader("Tornadodiagram Data")
        
        tornado_df = pd.DataFrame([
            {
                'Navn': row['Navn'],
                'Type': row['Type'],
                'P10 (%)': f"{(row['P10']/divisor):,.0f}".replace(",", " ") if row['Type'] != 'PNS' else '-',
                'P10 (kr)': f"{row['P10']:,.0f}".replace(",", " "),
                'P50 (kr)': f"{row['P50']:,.0f}".replace(",", " "),
                'P90 (%)': f"{(row['P90']/divisor):,.0f}".replace(",", " ") if row['Type'] != 'PNS' else '-',
                'P90 (kr)': f"{row['P90']:,.0f}".replace(",", " "),
                'Forventet (kr)': f"{row['Forventet verdi']:,.0f}".replace(",", " "),
                'Std.avvik': f"{row['Std.avvik']:,.0f}".replace(",", " "),
                'Spennvidde': f"{row['Spennvidde']:,.0f}".replace(",", " ")
            }
            for _, row in df_tornado.sort_values('Spennvidde', ascending=False).iterrows()
        ])
        
        st.dataframe(tornado_df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("📭 Kjør simulering i '⚙️ Oppsett'-fanen først")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
**Usikkerhetsanalyse Simulator** | Beta-PERT fordeling | Multiplikator-baserte usikkerhetsdrivere
- 💾 Data lagres lokalt i `usikkerhet_data.json`
- 📊 Beta-PERT-fordeling
- 🎲 Uavhengige simuleringer
""")