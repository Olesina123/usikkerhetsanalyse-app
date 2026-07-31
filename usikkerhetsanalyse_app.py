"""
Usikkerhetsanalyse - interaktiv web-app (Streamlit)

Kjør lokalt:
    pip install streamlit numpy pandas scipy matplotlib
    streamlit run usikkerhetsanalyse_app.py

Deployer gratis for andre (URL de kan besøke):
    1. Last opp denne filen (+ requirements.txt) til et GitHub-repo
    2. Gå til streamlit.io/cloud -> "New app" -> koble til repoet
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Usikkerhetsanalyse", layout="wide")
rng = np.random.default_rng(seed=42)


# ---------------------------------------------------------------
# Kjernefunksjoner (samme logikk som notebooken)
# ---------------------------------------------------------------

def fit_pert_alt(mode, p10_val, p90_val, lam=4):
    def equations(ab):
        a, b = ab
        if b <= a or not (a <= mode <= b):
            return [1e6, 1e6]
        alpha = 1 + lam * (mode - a) / (b - a)
        beta_p = 1 + lam * (b - mode) / (b - a)
        cdf10 = beta.cdf((p10_val - a) / (b - a), alpha, beta_p)
        cdf90 = beta.cdf((p90_val - a) / (b - a), alpha, beta_p)
        return [cdf10 - 0.10, cdf90 - 0.90]

    span = p90_val - p10_val if p90_val != p10_val else 1
    a0 = p10_val - 0.3 * span
    b0 = p90_val + 0.3 * span
    a, b = fsolve(equations, [a0, b0])
    return a, b


def riskpertalt_simuler(p10, mest_sannsynlig, p90, n=100_000):
    if p10 == mest_sannsynlig == p90:
        return np.full(n, p10)
    a, b = fit_pert_alt(mest_sannsynlig, p10, p90)
    alpha = 1 + 4 * (mest_sannsynlig - a) / (b - a)
    beta_param = 1 + 4 * (b - mest_sannsynlig) / (b - a)
    return a + (b - a) * beta.rvs(alpha, beta_param, size=n, random_state=rng)


# ---------------------------------------------------------------
# Standardverdier (samme PNS-poster og usikkerhetsdrivere som notebooken)
# ---------------------------------------------------------------

if "pns_df" not in st.session_state:
    st.session_state.pns_df = pd.DataFrame([
        {"navn": "PNS1 Entreprise – fast pris", "basisestimat": 72_759_975, "p10": 72_759_975, "mest_sannsynlig": 72_759_975, "p90": 72_759_975},
        {"navn": "PNS2 Grunnarbeid på land", "basisestimat": 6_006_672, "p10": 5_826_472, "mest_sannsynlig": 6_006_672, "p90": 6_246_939},
        {"navn": "PNS3 Mudringsarbeider", "basisestimat": 0, "p10": 0, "mest_sannsynlig": 0, "p90": 0},
        {"navn": "PNS4 Forurensede masser", "basisestimat": 3_342_006, "p10": 3_007_805, "mest_sannsynlig": 3_342_006, "p90": 4_611_968},
        {"navn": "PNS5 Sprengning", "basisestimat": 48_444, "p10": 48_444, "mest_sannsynlig": 48_444, "p90": 48_444},
        {"navn": "PNS6 Utfyllingsarbeider", "basisestimat": 625_000, "p10": 193_750, "mest_sannsynlig": 625_000, "p90": 1_918_750},
        {"navn": "PNS7 Plastring", "basisestimat": 106_260, "p10": 106_260, "mest_sannsynlig": 106_260, "p90": 198_706},
        {"navn": "PNS8 Endringsmeldinger", "basisestimat": 19_047_692, "p10": 16_190_538, "mest_sannsynlig": 19_047_692, "p90": 35_238_230},
        {"navn": "PNS9 Mengde og enhetspriser", "basisestimat": 486_861, "p10": 438_175, "mest_sannsynlig": 486_861, "p90": 535_547},
        {"navn": "PNS10 Byggherrekostnader", "basisestimat": 11_886_112, "p10": 10_697_501, "mest_sannsynlig": 11_886_112, "p90": 13_669_029},
        {"navn": "PNS11 Rigg og drift regulering", "basisestimat": 2_307_397, "p10": 2_307_397, "mest_sannsynlig": 2_307_397, "p90": 7_591_336},
    ])

if "u_df" not in st.session_state:
    st.session_state.u_df = pd.DataFrame([
        {"nr": "U2", "navn": "Lokale forhold", "p10": 1.00, "p50": 1.00, "p90": 1.041},
        {"nr": "U3", "navn": "Interessenter", "p10": 1.00, "p50": 1.00, "p90": 1.004},
        {"nr": "U4", "navn": "Prosjektutvikling, modenhet, løsning og omfang", "p10": 0.99, "p50": 1.057, "p90": 1.098},
        {"nr": "U5", "navn": "Offentlige godkjenninger og krav", "p10": 0.992, "p50": 1.00, "p90": 1.004},
        {"nr": "U6", "navn": "Entreprenør", "p10": 1.00, "p50": 1.00, "p90": 1.004},
        {"nr": "U7", "navn": "Prosjektledelse og eierstyring", "p10": 0.996, "p50": 1.00, "p90": 1.008},
    ])

if "palopt" not in st.session_state:
    st.session_state.palopt = 277_803_369
if "mva" not in st.session_state:
    st.session_state.mva = 0
if "n_sim" not in st.session_state:
    st.session_state.n_sim = 100_000


# ---------------------------------------------------------------
# UI - inputseksjon
# ---------------------------------------------------------------

st.title("Usikkerhetsanalyse")

st.subheader("PNS-poster (kalkyleposter)")
st.caption("Rediger tall direkte i tabellen. Bruk +/- nederst i tabellen for å legge til eller fjerne rader.")
pns_edit = st.data_editor(st.session_state.pns_df, num_rows="dynamic", use_container_width=True, key="pns_editor")

st.subheader("Usikkerhetsdrivere (U2-U7)")
u_edit = st.data_editor(st.session_state.u_df, num_rows="dynamic", use_container_width=True, key="u_editor")

col1, col2, col3 = st.columns(3)
with col1:
    palopt_input = st.number_input("Påløpt (fast, kr)", value=int(st.session_state.palopt), step=100_000)
with col2:
    mva_input = st.number_input("MVA (fast, kr)", value=int(st.session_state.mva), step=100_000)
with col3:
    n_sim_input = st.number_input("Antall simuleringer", value=int(st.session_state.n_sim), step=10_000, min_value=1_000)

simuler = st.button("Simuler", type="primary")


# ---------------------------------------------------------------
# Beregning - kjøres når "Simuler" trykkes
# ---------------------------------------------------------------

if simuler:
    pns_liste = pns_edit.to_dict("records")
    usikkerheter = u_edit.to_dict("records")

    # --- Simuler hver PNS-post ---
    resultater = []
    pns_samples_liste = []
    for p in pns_liste:
        samples = riskpertalt_simuler(p["p10"], p["mest_sannsynlig"], p["p90"], n=int(n_sim_input))
        pns_samples_liste.append(samples)
        resultater.append({
            "PNS": p["navn"],
            "Basisestimat": p["basisestimat"],
            "Forventningsverdi": round(np.mean(samples)),
            "Standard-avvik": round(np.std(samples)),
        })
    df_pns = pd.DataFrame(resultater)

    basiskostnad = df_pns["Basisestimat"].sum() + palopt_input
    basiskostnad_forv = df_pns["Forventningsverdi"].sum() + palopt_input

    # --- Simuler usikkerhetsdrivere (beregningsgrunnlag = sum av alle PNS forventningsverdier) ---
    beregningsgrunnlag = df_pns["Forventningsverdi"].sum()

    u_resultater = []
    u_samples_liste = []
    for u in usikkerheter:
        p10_effekt = beregningsgrunnlag * (u["p10"] - 1)
        p50_effekt = beregningsgrunnlag * (u["p50"] - 1)
        p90_effekt = beregningsgrunnlag * (u["p90"] - 1)
        samples = riskpertalt_simuler(p10_effekt, p50_effekt, p90_effekt, n=int(n_sim_input))
        u_samples_liste.append(samples)
        u_resultater.append({
            "Usikkerhet": u["nr"],
            "Navn": u["navn"],
            "Forventningsverdi": round(np.mean(samples)),
            "Standard-avvik": round(np.std(samples)),
        })
    df_u = pd.DataFrame(u_resultater)

    # --- Total prosjektkostnad (Monte Carlo - alle komponenter samtidig) ---
    total_samples = (
        np.sum(pns_samples_liste, axis=0)
        + palopt_input
        + mva_input
        + np.sum(u_samples_liste, axis=0)
    )
    total_mnok = total_samples / 1_000_000

    # --- Visning ---
    st.divider()
    st.subheader("Resultater: PNS-poster")
    st.dataframe(df_pns.style.format({"Basisestimat": "{:,.0f}", "Forventningsverdi": "{:,.0f}", "Standard-avvik": "{:,.0f}"}), use_container_width=True)

    st.subheader("Resultater: usikkerhetsdrivere")
    st.dataframe(df_u.style.format({"Forventningsverdi": "{:,.0f}", "Standard-avvik": "{:,.0f}"}), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Basiskostnad eks MVA/LPS", f"{basiskostnad_forv:,.0f} kr")
    c2.metric("P50 totalkostnad", f"{np.percentile(total_mnok, 50):.1f} MNOK")
    c3.metric("P85 totalkostnad", f"{np.percentile(total_mnok, 85):.1f} MNOK")

    st.subheader("S-kurve")
    sortert = np.sort(total_mnok)
    kumulativ = np.arange(1, len(sortert) + 1) / len(sortert) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sortert, kumulativ, color="#4a90d9", linewidth=2)
    ax.set_xlabel("Totalkostnad (MNOK)")
    ax.set_ylabel("Sannsynlighet (%)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Velg persentil (Px)")
    px_valgt = st.slider("Px", 1, 99, 50)
    st.write(f"P{px_valgt} = {np.percentile(total_mnok, px_valgt):.1f} MNOK ({np.percentile(total_mnok, px_valgt) * 1_000_000:,.0f} kr)")

else:
    st.info("Rediger tabellene over og trykk 'Simuler' for å kjøre analysen.")
