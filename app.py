"""
app.py — Sport AI Lab Start · l'interfaccia

Laboratorio didattico di machine learning per chinesiologi e nutrizionisti.
Tutta l'interfaccia Streamlit sta qui; i calcoli sono in ml.py e rl.py, i testi
didattici in didattica.py.

    1. STRUMENTI COMUNI          stile e piccole funzioni dell'interfaccia
    2. KPI SPORT E NUTRIZIONE    genera → analizza e pulisci → modelli ML
    3. I TUOI DATI               carica → analizza e pulisci → modelli ML
    4. RL CALCIO E NUTRIZIONE    bonus e malus → stato → allena → episodio
    5. HOME E NAVIGAZIONE        la barra laterale e la pagina iniziale

Avvio:
    python avvia.py                     (su Windows: doppio clic su avvia.bat)

Copyright (C) 2026  Christian Fiorino — licenza MIT
"""
from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Sport AI Lab Start", page_icon="🧪", layout="wide")

import matplotlib  # noqa: E402  (dopo set_page_config)
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import didattica as T  # noqa: E402
import ml  # noqa: E402
import rl  # noqa: E402


# ###########################################################################
# 1 · STRUMENTI COMUNI
# ###########################################################################
# comune.py — Sport AI Lab Start · piccoli strumenti condivisi dall'interfaccia

_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())


def _largo() -> dict:
    """`use_container_width` è stato sostituito da `width` nella 1.50."""
    try:
        ok = "width" in inspect.signature(st.button).parameters
    except (TypeError, ValueError):
        ok = False
    return {"width": "stretch"} if (_V >= (1, 50) and ok) else {"use_container_width": True}


LARGO = _largo()

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  div[data-testid="stMetricValue"] {font-size: 1.35rem;}
  .stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 0.4rem 0.9rem;}
  .intro {background:#eef4f3; border-left:4px solid #1b5e5a; padding:0.75rem 1.05rem;
          border-radius:4px; font-size:0.92rem; margin:0.3rem 0 1rem 0; color:#22303b;}
  .scheda {background:#fbfaf8; border:1px solid #e4e0d8; border-radius:8px;
           padding:0.9rem 1.15rem; margin-bottom:0.7rem;}
  .scheda h4 {margin:0 0 0.4rem 0; font-size:1.02rem; color:#22303b;}
  .scheda p {margin:0.22rem 0; font-size:0.9rem; color:#3d4855;}
  .lettura {background:#fdf6ef; border-left:4px solid #c2603a; padding:0.85rem 1.1rem;
            border-radius:4px; font-size:0.93rem; color:#22303b; margin:0.6rem 0 1rem 0;}
  .porta {background:#ffffff; border:1px solid #e4e0d8; border-radius:10px;
          padding:1.3rem 1.4rem; margin-bottom:0.8rem; min-height:16rem; color:#3d4855;}
  .porta h3 {margin:0 0 0.2rem 0; font-size:1.2rem; color:#1b5e5a;}
  .porta .occhiello {font-size:0.8rem; color:#8a9199; text-transform:uppercase;
          letter-spacing:0.06em; margin-bottom:0.5rem;}
  .porta p, .porta li {font-size:0.89rem; margin:0.3rem 0;}
  .porta ul {margin:0.4rem 0 0 1.1rem; padding:0;}
</style>
"""


def mostra(fig) -> None:
    """Disegna e chiude subito la figura (altrimenti la memoria cresce a ogni clic)."""
    if fig is None:
        return
    st.pyplot(fig)
    plt.close(fig)


def intro(testo: str) -> None:
    st.markdown(f'<div class="intro">{testo}</div>', unsafe_allow_html=True)


def lettura(righe) -> None:
    st.markdown(f'<div class="lettura">{"<br>".join(righe)}</div>', unsafe_allow_html=True)


def scheda(titolo: str, righe: list) -> None:
    corpo = "".join(f"<p>{r}</p>" for r in righe)
    st.markdown(f'<div class="scheda"><h4>{titolo}</h4>{corpo}</div>', unsafe_allow_html=True)


def a_json(d: pd.DataFrame) -> bytes:
    return json.dumps(json.loads(d.to_json(orient="records", force_ascii=False)),
                      indent=2, ensure_ascii=False).encode("utf-8")


def scarica_json(d: pd.DataFrame, nome_file: str, etichetta: str, chiave: str) -> None:
    st.download_button(etichetta, a_json(d), nome_file, "application/json", key=chiave)


# ###########################################################################
# 2 · KPI SPORT E KPI NUTRIZIONE
# ###########################################################################
# sezione_kpi.py — Sport AI Lab Start · sezioni KPI Sport e KPI Nutrizione
#
# Le due sezioni hanno lo stesso percorso in tre passi, cambia solo il dataset:
#
#     1 · Genera il dataset
#     2 · Analizza e pulisci
#     3 · Modelli di machine learning
#     📚 Didattica

TESTI = {
    "sport": {
        "icona": "🥅", "titolo": "KPI Sport",
        "contesto": "Una squadra di calcio seguita per una stagione: carico esterno dal GPS, "
                    "fatica percepita, sonno, HRV, benessere, prestazione e infortuni. "
                    "Una riga = un giocatore in una settimana.",
        "chi": "Giocatori", "uno": "giocatore", "periodo": "Settimane", "n": (16, 30, 22), "p": (8, 38, 20),
        "kmeans": ["distanza_km", "distanza_alta_int_m", "sprint_n", "vel_max_kmh"],
    },
    "nutrizione": {
        "icona": "🥗", "titolo": "KPI Nutrizione",
        "contesto": "Un gruppo di atleti di cinque discipline seguito giorno per giorno: diario "
                    "alimentare, colazione, sensore glicemico, dispendio, sonno, recupero e "
                    "disponibilità energetica. Una riga = un atleta in un giorno.",
        "chi": "Atleti", "uno": "atleta", "periodo": "Giorni", "n": (12, 32, 20), "p": (14, 60, 28),
        "kmeans": ["cho_g_kg", "pro_g_kg", "disponibilita_energetica", "picco_glicemico"],
    },
}


def _k(dom: str, nome: str) -> str:
    return f"{dom}_{nome}"


def rendi_kpi(dom: str) -> None:
    t = TESTI[dom]
    ss = st.session_state
    for nome in ("grezzo", "pulito", "registro", "ris", "confronto"):
        ss.setdefault(_k(dom, nome), None)

    st.title(f"{t['icona']} {t['titolo']}")
    intro(t["contesto"])

    tab1, tab2, tab3, tab4 = st.tabs(["1 · Genera il dataset", "2 · Analizza e pulisci",
                                      "3 · Modelli ML", "📚 Didattica"])

    # ================================================================ 1. GENERA
    with tab1:
        st.subheader("Genera il dataset")
        c1, c2, c3 = st.columns(3)
        n = c1.slider(t["chi"], *t["n"], key=_k(dom, "n"))
        periodo = c2.slider(t["periodo"], *t["p"], key=_k(dom, "p"))
        seme = c3.number_input("Variante (seme casuale)", 1, 999, 7, key=_k(dom, "seme"),
                               help="Stessa variante = stesso dataset. Assegnane una diversa "
                                    "a ogni studente.")
        errori = st.toggle("Inserisci errori realistici da pulire (consigliato)", True,
                           key=_k(dom, "err"))
        if errori:
            e1, e2, e3 = st.columns(3)
            q_vuote = e1.slider("% celle vuote", 0, 15, 4, key=_k(dom, "qv")) / 100
            q_imp = e2.slider("% valori impossibili", 0, 5, 2, key=_k(dom, "qi")) / 100
            q_dup = e3.slider("% righe duplicate", 0, 10, 2, key=_k(dom, "qd")) / 100

        if st.button("🎲 Genera il dataset", type="primary", key=_k(dom, "gen"), **LARGO):
            df = ml.genera(dom, n, periodo, int(seme))
            if errori:
                df = ml.sporca(df, dom, q_vuote, q_imp, q_dup, seed=int(seme))
            ss[_k(dom, "grezzo")] = df
            ss[_k(dom, "pulito")] = None
            ss[_k(dom, "registro")] = None
            ss[_k(dom, "ris")] = None
            ss[_k(dom, "confronto")] = None

        df = ss[_k(dom, "grezzo")]
        if df is None:
            st.info("Imposta i parametri e premi **Genera il dataset**.")
        else:
            g = ml.ID_GRUPPO[dom]
            m = st.columns(4)
            m[0].metric("Righe", len(df))
            m[1].metric("Colonne", df.shape[1])
            m[2].metric(t["chi"], df[g].nunique())
            m[3].metric("Celle vuote", int(df.isna().sum().sum()))
            st.dataframe(df.head(50), **LARGO, height=320)
            st.caption("Prime 50 righe. Il significato di ogni colonna è al passo 2.")
            scarica_json(df, f"{dom}_dataset_grezzo.json", "⬇️ Scarica il dataset (JSON)",
                         _k(dom, "dl_raw"))

    # ======================================================= 2. ANALIZZA E PULISCI
    with tab2:
        df = ss[_k(dom, "grezzo")]
        if df is None:
            st.warning("Prima genera il dataset al passo 1.")
        else:
            st.subheader("Che cosa significa ogni colonna")
            st.dataframe(ml.tabella_significato(dom, df), hide_index=True, **LARGO, height=420)
            st.caption("**Ruolo**: *id* = identifica la riga, non si usa nei modelli · *categoria* "
                       "= testo · *misura* = variabile in ingresso · *bersaglio* = ciò che si "
                       "può prevedere · *trappola* = nota solo dopo l'evento, non va mai usata.")

            st.divider()
            st.subheader("Pulizia del dataset")
            diag = ml.diagnosi(df, dom)
            dup = int(df.duplicated().sum())
            c = st.columns(3)
            c[0].metric("Righe duplicate", dup)
            c[1].metric("Valori impossibili", int(diag["valori impossibili"].sum()))
            c[2].metric("Celle vuote", int(diag["celle vuote"].sum()))
            problemi = diag[(diag["celle vuote"] > 0) | (diag["valori impossibili"] > 0)]
            if len(problemi):
                with st.expander("Dove sono i problemi, colonna per colonna"):
                    st.dataframe(problemi, hide_index=True, **LARGO)

            p1, p2, p3 = st.columns(3)
            togli = p1.checkbox("Elimina le righe duplicate", True, key=_k(dom, "dup"))
            imp = p2.selectbox("Valori impossibili", ml.OPZIONI_IMPOSSIBILI, key=_k(dom, "imp"))
            vuo = p3.selectbox("Celle vuote", ml.OPZIONI_MANCANTI, key=_k(dom, "vuo"))

            if st.button("🧹 Pulisci il dataset", type="primary", key=_k(dom, "pul"), **LARGO):
                pul, reg = ml.pulisci(df, dom, togli, imp, vuo)
                ss[_k(dom, "pulito")] = pul
                ss[_k(dom, "registro")] = reg
                ss[_k(dom, "ris")] = None
                ss[_k(dom, "confronto")] = None

            pul = ss[_k(dom, "pulito")]
            if pul is not None:
                st.success(f"Dataset pulito: {len(pul)} righe (erano {len(df)}). "
                           f"Celle vuote rimaste: {int(pul.isna().sum().sum())}. "
                           f"Valori impossibili rimasti: "
                           f"{int(ml.diagnosi(pul, dom)['valori impossibili'].sum())}.")
                st.markdown("**Registro della pulizia**")
                st.dataframe(ss[_k(dom, "registro")], hide_index=True, **LARGO)
                a, b = st.columns(2)
                with a:
                    scarica_json(pul, f"{dom}_dataset_pulito.json",
                                 "⬇️ Scarica il dataset pulito (JSON)", _k(dom, "dl_pul"))
                with b:
                    scarica_json(ss[_k(dom, "registro")], f"{dom}_registro_pulizia.json",
                                 "⬇️ Scarica il registro (JSON)", _k(dom, "dl_reg"))

    # ======================================================= 3. MODELLI
    with tab3:
        df = ss[_k(dom, "pulito")]
        if ss[_k(dom, "grezzo")] is None:
            st.warning("Prima genera il dataset al passo 1.")
        else:
            if df is None:
                df = ss[_k(dom, "grezzo")]
                st.warning("Stai usando il dataset **non pulito**. Le celle vuote verranno "
                           "riempite in automatico, ma duplicati e valori impossibili restano. "
                           "Meglio passare dal passo 2.")
            _sezione_modelli(dom, df, int(ss.get(_k(dom, "seme"), 7)))

    # ======================================================= DIDATTICA
    with tab4:
        _didattica(dom, ss[_k(dom, "grezzo")])


# =========================================================================
# PASSO 3
# =========================================================================

def _sezione_modelli(dom: str, df: pd.DataFrame, seme: int) -> None:
    ss = st.session_state
    st.subheader("Scegli il modello")
    nomi = list(ml.MODELLI)
    modello = st.radio("Modello", nomi, horizontal=True, key=_k(dom, "mod"),
                       label_visibility="collapsed")
    scheda(modello, [T.COSA_FA[modello]])
    st.markdown(T.USO[dom][modello])
    if modello == "XGBoost" and not ml.XGB_OK:
        st.info("La libreria `xgboost` non è installata: al suo posto viene usato il "
                "Gradient Boosting di scikit-learn, che segue lo stesso principio. "
                "Per la versione originale: `pip install xgboost`.")

    st.divider()

    # ---------------------------------------------------------- K-MEANS
    if modello == "K-Means":
        prof = ml.profili_medi(df, dom)
        numeriche = [c for c in prof.columns
                     if c != ml.ID_GRUPPO[dom] and pd.api.types.is_numeric_dtype(prof[c])
                     and c not in ("settimana", "giorno")]
        default = [c for c in TESTI[dom]["kmeans"] if c in numeriche]
        st.caption(f"Il K-Means lavora sul **profilo medio** di ogni {TESTI[dom]['uno']}: "
                   f"una riga per persona ({len(prof)} in tutto), con la media di ogni variabile.")
        variabili = st.multiselect("Variabili su cui cercare i gruppi", numeriche, default,
                                   key=_k(dom, "kvar"))
        k = st.slider("Numero di gruppi (k)", 2, 7, 3, key=_k(dom, "kk"))
        if st.button("▶️ Esegui K-Means", type="primary", key=_k(dom, "run_k"), **LARGO):
            try:
                ss[_k(dom, "ris")] = ml.raggruppa(df, dom, variabili, k, seed=seme)
            except Exception as e:
                st.error(str(e))
        ris = ss[_k(dom, "ris")]
        if ris is not None and ris["compito"] == "raggruppamento":
            _risultato_kmeans(dom, ris)
        return

    # ---------------------------------------------------------- SUPERVISIONATI
    possibili = ml.bersagli_per(modello, dom)
    cat = ml.catalogo(dom)
    c1, c2 = st.columns([1, 1])
    bersaglio = c1.selectbox("Variabile bersaglio (target)", possibili,
                             format_func=lambda b: f"{b} — {cat[b].unita}", key=_k(dom, f"b_{modello}"))
    c2.markdown(f"<br><small>{cat[bersaglio].descrizione}</small>", unsafe_allow_html=True)

    ammessi = ml.predittori(dom, df, bersaglio)
    escluse = ml.ESCLUSE_PER_BERSAGLIO.get(bersaglio, {})
    predittori = st.multiselect("Variabili in ingresso", ammessi, ammessi,
                                key=_k(dom, f"x_{bersaglio}"))
    if escluse:
        testo = ("1 colonna esclusa" if len(escluse) == 1 else f"{len(escluse)} colonne escluse")
        with st.expander(f"🚫 {testo} in automatico (contengono la risposta)"):
            for col, perche in escluse.items():
                st.markdown(f"- **{col}** — {perche}")

    compito = ml.compito_di(dom, bersaglio)
    par = {}
    soglia = 0.5
    with st.expander("Impostazioni (facoltative)"):
        quota = st.slider(f"Quota di {TESTI[dom]['chi'].lower()} tenuti da parte per la verifica",
                          0.15, 0.40, 0.25, 0.05, key=_k(dom, "quota"))
        if modello in ("Random Forest", "XGBoost"):
            par["alberi"] = st.slider("Numero di alberi", 50, 600, 300, 50, key=_k(dom, "alb"))
            prof = st.slider("Profondità massima degli alberi (0 = libera)", 0, 12,
                             0 if modello == "Random Forest" else 3, key=_k(dom, f"pr_{modello}"))
            par["profondita"] = prof or None
        if modello == "XGBoost":
            par["ritmo"] = st.slider("Velocità di apprendimento", 0.01, 0.3, 0.05, 0.01,
                                     key=_k(dom, "lr"))
    if compito == "classificazione":
        soglia = st.slider("Soglia di allarme (probabilità oltre la quale il modello dice «sì»)",
                           0.05, 0.95, 0.30, 0.05, key=_k(dom, "soglia"))

    a, b = st.columns(2)
    if a.button(f"▶️ Esegui {modello}", type="primary", key=_k(dom, "run"), **LARGO):
        try:
            with st.spinner("Addestramento…"):
                ss[_k(dom, "ris")] = ml.addestra(df, dom, modello, bersaglio, predittori, seed=seme,
                                                quota=quota, soglia=soglia, par=par)
        except Exception as e:
            st.error(str(e))
    if b.button("⚖️ Confronta tutti i modelli su questo bersaglio", key=_k(dom, "cmp"), **LARGO):
        try:
            with st.spinner("Un addestramento per modello…"):
                ss[_k(dom, "confronto")] = (bersaglio, ml.confronta_tutti(df, dom, bersaglio,
                                                                         predittori, seed=seme,
                                                                         quota=quota))
        except Exception as e:
            st.error(str(e))

    ris = ss[_k(dom, "ris")]
    if ris is not None and ris["compito"] != "raggruppamento":
        st.divider()
        _risultato_supervisionato(dom, ris)

    cmp = ss[_k(dom, "confronto")]
    if cmp is not None:
        b_cmp, tab = cmp
        st.divider()
        st.subheader(f"Confronto fra modelli · bersaglio `{b_cmp}`")
        st.dataframe(tab, hide_index=True, **LARGO)
        mostra(ml.g_confronto(tab))
        validi = tab[~tab["modello"].str.contains("riferimento")]
        migliore = validi.loc[validi["R² verifica"].idxmax()]
        lettura([f"Il modello con l'R² in verifica più alto è <b>{migliore['modello']}</b> "
                 f"(R² = {migliore['R² verifica']:.2f}, MAE = {migliore['MAE verifica']:.2f}).",
                 "Se la differenza con la Regressione Lineare/Logistica è piccola, preferisci "
                 "il modello più semplice: si spiega meglio e sbaglia in modo più prevedibile.",
                 "Guarda anche la distanza fra R² di addestramento e di verifica: più è grande, "
                 "più il modello ha imparato a memoria."])
        scarica_json(tab, f"{dom}_confronto_{b_cmp}.json", "⬇️ Scarica il confronto (JSON)",
                     _k(dom, "dl_cmp"))


def _risultato_supervisionato(dom: str, ris: dict) -> None:
    cat = ml.catalogo(dom)
    unita = cat[ris["bersaglio"]].unita
    st.subheader(f"Risultato · {ris['modello']} su `{ris['bersaglio']}`")
    m, rif = ris["misure"], ris["riferimento"]
    c = st.columns(4)
    c[0].metric("R² (verifica)", f"{m['R² (verifica)']:.3f}",
                delta=f"{m['R² (verifica)'] - rif['R² (verifica)']:+.3f} sulla media")
    c[1].metric(f"MAE (verifica){'' if ris['compito'] == 'classificazione' else ' · ' + unita}",
                f"{m['MAE (verifica)']:.3f}",
                delta=f"{m['MAE (verifica)'] - rif['MAE (verifica)']:+.3f} sulla media",
                delta_color="inverse")
    c[2].metric("R² (addestramento)", f"{m['R² (addestramento)']:.3f}")
    c[3].metric("MAE (addestramento)", f"{m['MAE (addestramento)']:.3f}")
    if ris["compito"] == "classificazione":
        c2 = st.columns(4)
        c2[0].metric("AUC", f"{m['AUC']:.3f}")
        c2[1].metric("Sensibilità", f"{m['Sensibilità (eventi riconosciuti)'] * 100:.0f}%")
        c2[2].metric("Specificità", f"{m['Specificità (non-eventi riconosciuti)'] * 100:.0f}%")
        c2[3].metric("Accuratezza", f"{m['Accuratezza'] * 100:.0f}%")
    st.caption(f"Addestrato su {ris['n_addestramento']} righe, verificato su {ris['n_verifica']} "
               f"righe di {len(ris['atleti_verifica'])} persone mai viste: "
               f"{', '.join(ris['atleti_verifica'])}.")

    lettura(ml.commenta(ris, "" if ris["compito"] == "classificazione" else unita))

    a, b = st.columns(2)
    with a:
        if ris["compito"] == "regressione":
            mostra(ml.g_osservato_stimato(ris))
            st.caption("Ogni punto è una riga di verifica. Più i punti stanno sulla diagonale, "
                       "più il modello è preciso.")
        else:
            mostra(ml.g_confusione(ris))
            st.caption("In basso a destra gli eventi presi, in basso a sinistra quelli mancati, "
                       "in alto a destra i falsi allarmi.")
    with b:
        fig = ml.g_importanza(ris)
        if fig is not None:
            mostra(fig)
            st.caption("Per gli alberi: quanto ogni variabile è servita a dividere i dati. Per i "
                       "modelli lineari: grandezza del coefficiente sui dati standardizzati.")
    if ris["compito"] == "classificazione":
        a, _ = st.columns(2)
        with a:
            mostra(ml.g_probabilita(ris))
            st.caption("Le barre arancioni (eventi veri) dovrebbero stare a destra della soglia, "
                       "quelle grigie a sinistra. Sposta la soglia e riesegui per vedere l'effetto.")
    if ris.get("coefficienti") is not None:
        with st.expander("I coefficienti del modello lineare"):
            st.dataframe(ris["coefficienti"], hide_index=True, **LARGO)
            st.caption("Calcolati su dati standardizzati: si confrontano fra loro. Un coefficiente "
                       "di 0,5 vuol dire che una deviazione standard in più della variabile sposta "
                       "il bersaglio di 0,5 unità.")
    with st.expander("Le previsioni, riga per riga"):
        st.dataframe(ris["tabella"], hide_index=True, **LARGO, height=300)

    pacchetto = {
        "dataset": dom, "modello": ris["modello"], "bersaglio": ris["bersaglio"],
        "compito": ris["compito"], "variabili_in_ingresso": ris["predittori"],
        "misure": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in m.items()},
        "riferimento_media": {k: round(float(v), 4) for k, v in rif.items()},
        "persone_in_verifica": ris["atleti_verifica"],
        "importanza": (ris["importanza"].to_dict(orient="records")
                       if ris["importanza"] is not None else None),
        "previsioni": json.loads(ris["tabella"].to_json(orient="records")),
        "prodotto_il": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.download_button("⬇️ Scarica il risultato completo (JSON)",
                        json.dumps(pacchetto, indent=2, ensure_ascii=False).encode("utf-8"),
                        f"{dom}_{ris['modello'].replace(' ', '_')}_{ris['bersaglio']}.json",
                        "application/json", key=_k(dom, "dl_ris"))


def _risultato_kmeans(dom: str, ris: dict) -> None:
    st.divider()
    st.subheader(f"Risultato · K-Means con k = {ris['k']}")
    m = ris["misure"]
    nomi = list(m)
    c = st.columns(2)
    c[0].metric("R² (varianza spiegata dai gruppi)", f"{m[nomi[0]]:.3f}")
    c[1].metric("MAE (distanza media dal centro, in dev. std.)", f"{m[nomi[1]]:.3f}")
    lettura(ml.commenta(ris))
    a, b = st.columns(2)
    with a:
        mostra(ml.g_mappa_gruppi(ris))
        st.caption("Le variabili scelte riassunte in due dimensioni (PCA). Punti vicini = "
                   "persone simili.")
    with b:
        mostra(ml.g_gomito(ris))
        st.caption("Si sceglie il k dove la curva smette di salire in fretta.")
    st.markdown("**Identikit dei gruppi** (media di ogni variabile)")
    st.dataframe(ris["identikit"], hide_index=True, **LARGO)
    if ris["composizione"] is not None:
        st.markdown(f"**Composizione per {ris['categoria']}**")
        st.dataframe(ris["composizione"], **LARGO)
    with st.expander("Il gruppo di ogni persona"):
        g = ml.ID_GRUPPO[dom]
        col = [g] + ([ris["categoria"]] if ris["categoria"] in ris["tabella"].columns else []) \
            + ["gruppo"] + list(ris["variabili"])
        st.dataframe(ris["tabella"][col], hide_index=True, **LARGO)
    pacchetto = {
        "dataset": dom, "modello": "K-Means", "k": ris["k"], "variabili": ris["variabili"],
        "misure": {k: round(float(v), 4) for k, v in m.items()},
        "gomito": json.loads(ris["gomito"].round(4).to_json(orient="records")),
        "identikit": json.loads(ris["identikit"].to_json(orient="records")),
        "assegnazioni": json.loads(ris["tabella"].drop(columns=["pca_1", "pca_2"])
                                   .to_json(orient="records")),
    }
    st.download_button("⬇️ Scarica il risultato completo (JSON)",
                        json.dumps(pacchetto, indent=2, ensure_ascii=False).encode("utf-8"),
                        f"{dom}_kmeans_k{ris['k']}.json", "application/json",
                        key=_k(dom, "dl_km"))


# =========================================================================
# DIDATTICA
# =========================================================================

def _didattica(dom: str, df: pd.DataFrame | None) -> None:
    s1, s2, s3, s4, s5, s6 = st.tabs(["Il percorso", "Le colonne", "I modelli", "R² e MAE",
                                      "Pulizia dei dati", "Esercizi e glossario"])
    with s1:
        st.markdown(T.PERCORSO)
    with s2:
        st.dataframe(ml.tabella_significato(dom, df), hide_index=True, **LARGO, height=520)
        scarica_json(ml.tabella_significato(dom, df), f"{dom}_significato_colonne.json",
                     "⬇️ Scarica la scheda delle colonne (JSON)", _k(dom, "dl_cat"))
    with s3:
        st.markdown("#### Qual è il modello più adatto a ciascuno scopo")
        st.dataframe(pd.DataFrame(T.QUALE_SCOPO[dom], columns=["scopo", "modello consigliato",
                                                                "perché"]),
                     hide_index=True, **LARGO)
        st.markdown("#### Come funziona ogni modello e come si usa su questo dataset")
        for nome in ml.MODELLI:
            with st.expander(nome):
                st.markdown(T.COSA_FA[nome])
                st.markdown(T.USO[dom][nome])
        st.markdown("#### In sintesi")
        st.dataframe(pd.DataFrame([
            ["Regressione Lineare", "numero", "altissima", "bassa", "spiegare"],
            ["Regressione Logistica", "sì/no", "altissima", "bassa", "spiegare un rischio"],
            ["Random Forest", "numero o sì/no", "media", "media", "soglie e interazioni"],
            ["XGBoost", "numero o sì/no", "bassa", "alta", "massima precisione"],
            ["K-Means", "nessuno", "media", "—", "trovare gruppi"],
        ], columns=["modello", "bersaglio", "leggibilità", "rischio di sovradattamento",
                    "punto di forza"]), hide_index=True, **LARGO)
    with s4:
        st.markdown(T.METRICHE)
    with s5:
        st.markdown(T.PULIZIA)
    with s6:
        st.markdown("#### Esercizi")
        for i, (tit, cosa, consegna) in enumerate(T.ESERCIZI[dom], 1):
            scheda(f"{i}. {tit}", [f"<b>Che cosa fare</b>: {cosa}", f"<b>Consegna</b>: {consegna}"])
        st.markdown("#### Glossario")
        for termine, definizione in T.GLOSSARIO:
            st.markdown(f"**{termine}** — {definizione}")


# ###########################################################################
# 3 · I TUOI DATI
# ###########################################################################
# sezione_utente.py — Sport AI Lab Start · sezione «I tuoi dati»
#
# Lo stesso percorso delle sezioni KPI, su un dataset caricato dallo studente:
#
#     1 · Carica il dataset
#     2 · Analizza e pulisci
#     3 · Modelli ML
#     📚 Didattica

KU = "u_"


def _azzera(*chiavi: str) -> None:
    for c in chiavi:
        st.session_state[KU + c] = None


def rendi_utente() -> None:
    ss = st.session_state
    for c in ("grezzo", "note", "schema", "pulito", "registro", "schema_pulito", "ris",
              "confronto", "firma", "nome"):
        ss.setdefault(KU + c, None)

    st.title("📂 I tuoi dati")
    intro("Carica un tuo dataset — CSV, Excel o JSON — e segui lo stesso percorso delle altre "
          "sezioni: <b>capirlo, pulirlo, analizzarlo</b>. Le colonne possono essere qualsiasi: "
          "l'app riconosce tipi e ruoli, tu li correggi, e i modelli si adattano.")

    t1, t2, t3, t4 = st.tabs(["1 · Carica il dataset", "2 · Analizza e pulisci", "3 · Modelli ML",
                              "📚 Didattica"])

    # ================================================================ 1. CARICA
    with t1:
        file = st.file_uploader("Trascina qui il file", type=["csv", "txt", "tsv", "xlsx", "xls",
                                                              "xlsm", "json"], key=KU + "file")
        with st.expander("Opzioni di lettura (se la tabella non si legge bene)"):
            a, b, c = st.columns(3)
            sep = a.selectbox("Separatore (solo CSV)", ["auto", ";", ",", "\t", "|"],
                              format_func=lambda s: {"auto": "riconosci da solo",
                                                     "\t": "tabulazione"}.get(s, s),
                              key=KU + "sep")
            intest = b.number_input("Riga dell'intestazione (0 = la prima)", 0, 20, 0,
                                    key=KU + "intest")
            foglio = None
            if file is not None and file.name.lower().endswith(("xlsx", "xls", "xlsm")):
                try:
                    fogli = ml.fogli_excel(file.getvalue())
                    foglio = c.selectbox("Foglio Excel", fogli,
                                         index=ml.foglio_principale(file.getvalue()),
                                         key=KU + "foglio_" + file.name)
                except Exception as e:
                    st.error(f"Non riesco ad aprire il file Excel: {e}")

        a, b = st.columns([3, 1])
        a.caption("Non hai un file a portata di mano? Prova con un dataset di esempio, con "
                  "qualche errore da pulire.")
        esempio = b.button("🎲 Usa un esempio", key=KU + "esempio", **LARGO)

        if file is not None:
            firma = hashlib.md5(file.getvalue()).hexdigest() + f"{sep}{intest}{foglio}"
            if firma != ss[KU + "firma"]:
                try:
                    df, note = ml.leggi(file.getvalue(), file.name, foglio, sep, int(intest))
                    if df.empty or df.shape[1] < 2:
                        raise ValueError("la tabella letta ha meno di due colonne: prova a "
                                         "cambiare separatore o riga di intestazione.")
                    _nuovo(df, note, file.name, firma)
                except Exception as e:
                    _azzera("grezzo", "schema", "pulito", "registro", "schema_pulito", "ris",
                            "confronto")
                    ss[KU + "firma"] = firma
                    ss[KU + "errore"] = str(e)
            if ss[KU + "grezzo"] is None and ss.get(KU + "errore"):
                st.error(f"Non riesco a leggere il file: {ss[KU + 'errore']}")
        elif esempio:
            df = ml.sporca(ml.genera_sport(18, 16, 3), "sport", 0.05, 0.02, 0.02, 3)
            df = df.drop(columns=["giorni_stop"]).rename(columns={"id_giocatore": "codice"})
            df["note"] = np.where(np.random.default_rng(0).random(len(df)) < 0.1,
                                  "dolore al flessore", "")
            df, note = ml.converti_tipi(df)
            _nuovo(df, ["Dataset di esempio: una squadra di calcio, con errori da pulire."],
                   "esempio_squadra.csv", "esempio")

        df = ss[KU + "grezzo"]
        if df is None:
            st.info("Carica un file per iniziare.")
        else:
            st.success(f"**{ss[KU + 'nome']}** — {df.shape[0]} righe × {df.shape[1]} colonne.")
            with st.expander("Come è stato letto il file", expanded=False):
                for n in ss[KU + "note"]:
                    st.markdown(f"- {n}")
            st.dataframe(df.head(50), **LARGO, height=320)
            st.caption("Prime 50 righe. Se le colonne sono tutte in una, cambia il separatore "
                       "nelle opzioni di lettura.")
            scarica_json(df, "dataset_caricato.json", "⬇️ Scarica il dataset letto (JSON)",
                         KU + "dl_raw")

    # ================================================================ 2. ANALIZZA
    with t2:
        df = ss[KU + "grezzo"]
        if df is None:
            st.warning("Prima carica un dataset al passo 1.")
        else:
            st.subheader("Che cosa significa ogni colonna")
            st.caption("Correggi il **ruolo** dove serve e, se vuoi, scrivi significato, unità e "
                       "limiti plausibili. Le colonne grigie sono calcolate dall'app.")
            schema = st.data_editor(
                ss[KU + "schema"], key=KU + "editor_" + str(ss[KU + "firma"]), hide_index=True,
                **LARGO, height=min(460, 40 + 36 * len(ss[KU + "schema"])),
                disabled=["colonna", "tipo rilevato", "valori distinti", "celle vuote %",
                          "esempio", "min", "max"],
                column_config={
                    "ruolo": st.column_config.SelectboxColumn("ruolo", options=ml.RUOLI_COLONNA,
                                                              required=True),
                    "significato": st.column_config.TextColumn("significato", width="medium"),
                    "min plausibile": st.column_config.NumberColumn("min plausibile"),
                    "max plausibile": st.column_config.NumberColumn("max plausibile"),
                })
            ss[KU + "schema_corrente"] = schema
            ids = ml.colonne_per_ruolo(schema, "identificativo atleta")
            if len(ids) > 1:
                st.warning(f"Più colonne come identificativo ({', '.join(ids)}): userò solo "
                           f"«{ids[0]}».")
            r = schema["ruolo"].value_counts()
            c = st.columns(4)
            for col, ruolo in zip(c, ml.RUOLI_COLONNA):
                col.metric(ruolo.capitalize(), int(r.get(ruolo, 0)))
            scarica_json(ml.tabella_significato_utente(schema), "significato_colonne.json",
                         "⬇️ Scarica la scheda delle colonne (JSON)", KU + "dl_sch")

            st.divider()
            st.subheader("Pulizia del dataset")
            diag = ml.diagnosi_utente(df, schema)
            dup = int(df.duplicated().sum())
            c = st.columns(4)
            c[0].metric("Righe duplicate", dup)
            c[1].metric("Celle vuote", int(diag["celle vuote"].sum()))
            c[2].metric("Fuori dai limiti plausibili", int(diag["fuori dai limiti plausibili"].sum()))
            c[3].metric("Valori estremi (Tukey)", int(diag["valori estremi (Tukey)"].sum()))
            problemi = diag[(diag[["celle vuote", "fuori dai limiti plausibili",
                                   "valori estremi (Tukey)"]].sum(axis=1) > 0)]
            if len(problemi):
                with st.expander("Dove sono i problemi, colonna per colonna"):
                    st.dataframe(problemi, hide_index=True, **LARGO)
            st.caption("**Fuori dai limiti** usa i limiti che hai scritto nella tabella. **Valori "
                       "estremi** usa la regola di Tukey (oltre 3 scarti interquartili): trova "
                       "gli errori evidenti anche senza limiti, ma può colpire valori veri e rari.")

            a, b, c = st.columns(3)
            dupl = a.checkbox("Elimina le righe duplicate", True, key=KU + "dup")
            soglia_v = a.slider("Togli le colonne vuote oltre il (%)", 10, 100, 60, 5,
                                key=KU + "sogliav")
            fuori = b.selectbox("Valori fuori dai limiti", ml.OPZ_FUORI, key=KU + "fuori")
            estr = b.selectbox("Valori estremi (Tukey)", ml.OPZ_ESTREMI, key=KU + "estr")
            vn = c.selectbox("Celle vuote nei numeri", ml.OPZ_VUOTE_NUM, key=KU + "vn")
            vc = c.selectbox("Celle vuote nelle categorie", ml.OPZ_VUOTE_CAT, key=KU + "vc")
            if st.button("🧹 Pulisci il dataset", type="primary", key=KU + "pul", **LARGO):
                pul, reg, sch2 = ml.pulisci_utente(df, schema, dupl, float(soglia_v), fuori, estr, vn, vc)
                ss[KU + "pulito"], ss[KU + "registro"], ss[KU + "schema_pulito"] = pul, reg, sch2
                _azzera("ris", "confronto")

            if ss[KU + "pulito"] is not None:
                pul = ss[KU + "pulito"]
                usate = [c for c in ml.colonne_per_ruolo(ss[KU + "schema_pulito"], "misura",
                                                        "categoria") if c in pul.columns]
                st.success(f"Dataset pulito: {len(pul)} righe (erano {len(df)}), "
                           f"{pul.shape[1]} colonne. Celle vuote rimaste nelle colonne usate "
                           f"dai modelli: {int(pul[usate].isna().sum().sum())}.")
                st.dataframe(ss[KU + "registro"], hide_index=True, **LARGO)
                a, b = st.columns(2)
                with a:
                    scarica_json(pul, "dataset_pulito.json", "⬇️ Scarica il dataset pulito (JSON)",
                                 KU + "dl_pul")
                with b:
                    scarica_json(ss[KU + "registro"], "registro_pulizia.json",
                                 "⬇️ Scarica il registro (JSON)", KU + "dl_reg")
                st.caption("Se cambi ruoli o limiti nella tabella, premi di nuovo «Pulisci».")

    # ================================================================ 3. MODELLI
    with t3:
        if ss[KU + "grezzo"] is None:
            st.warning("Prima carica un dataset al passo 1.")
        else:
            dati = ss[KU + "pulito"]
            schema = ss[KU + "schema_pulito"]
            if dati is None:
                dati = ss[KU + "grezzo"]
                schema = ss.get(KU + "schema_corrente", ss[KU + "schema"])
                st.warning("Stai usando il dataset **non pulito**: le celle vuote vengono "
                           "riempite in automatico, ma duplicati ed errori restano. Meglio "
                           "passare dal passo 2.")
            _modelli(dati, schema)

    # ================================================================ DIDATTICA
    with t4:
        s1, s2, s3, s4 = st.tabs(["Preparare il file", "Ruoli delle colonne",
                                  "Come si adattano i modelli", "I modelli e R² / MAE"])
        with s1:
            st.markdown(T.PREPARA_FILE)
        with s2:
            st.markdown(T.RUOLI_SPIEGATI)
        with s3:
            st.markdown(T.ADATTAMENTO)
        with s4:
            for nome in ml.MODELLI:
                with st.expander(nome):
                    st.markdown(T.COSA_FA[nome])
                    st.markdown(T.USO_GENERICO[nome])
            st.markdown(T.METRICHE)


def _nuovo(df: pd.DataFrame, note, nome: str, firma: str) -> None:
    ss = st.session_state
    ss[KU + "errore"] = None
    ss[KU + "grezzo"], ss[KU + "note"], ss[KU + "nome"], ss[KU + "firma"] = df, note, nome, firma
    ss[KU + "schema"] = ml.profilo(df)
    ss[KU + "schema_corrente"] = ss[KU + "schema"]
    _azzera("pulito", "registro", "schema_pulito", "ris", "confronto")


# =========================================================================
# PASSO 3
# =========================================================================

def _modelli(dati: pd.DataFrame, schema: pd.DataFrame) -> None:
    ss = st.session_state
    gruppo = ml.colonna_id(schema)
    if gruppo not in dati.columns:
        gruppo = None
    candidate = [c for c in ml.colonne_per_ruolo(schema, "misura", "categoria") if c in dati.columns]
    categoriali = [c for c in ml.colonne_per_ruolo(schema, "categoria") if c in dati.columns]
    numeriche = [c for c in ml.colonne_per_ruolo(schema, "misura")
                 if c in dati.columns and pd.api.types.is_numeric_dtype(dati[c])]
    if len(candidate) < 2:
        st.error("Servono almeno due colonne con ruolo *misura* o *categoria* (passo 2).")
        return

    st.subheader("Scegli il modello")
    modello = st.radio("Modello", list(ml.MODELLI), horizontal=True, key=KU + "mod",
                       label_visibility="collapsed")
    scheda(modello, [T.COSA_FA[modello]])
    st.markdown(T.USO_GENERICO[modello])
    if modello == "XGBoost" and not ml.XGB_OK:
        st.info("`xgboost` non è installato: uso il Gradient Boosting di scikit-learn.")
    st.divider()

    # ---------------------------------------------------------- K-MEANS
    if modello == "K-Means":
        if len(numeriche) < 2:
            st.error("Servono almeno due colonne numeriche con ruolo *misura*.")
            return
        variabili = st.multiselect("Variabili su cui cercare i gruppi", numeriche,
                                   numeriche[:min(4, len(numeriche))], key=KU + "kvar")
        a, b, c = st.columns(3)
        k = a.slider("Numero di gruppi (k)", 2, 8, 3, key=KU + "kk")
        per_atleta = b.checkbox(f"Una riga per atleta (media per «{gruppo}»)", bool(gruppo),
                                key=KU + "kpa", disabled=gruppo is None)
        descr = c.selectbox("Categoria da confrontare con i gruppi (facoltativa)",
                            ["—"] + categoriali, key=KU + "kdesc")
        if st.button("▶️ Esegui K-Means", type="primary", key=KU + "runk", **LARGO):
            try:
                ss[KU + "ris"] = ml.raggruppa_generico(dati, variabili, k, gruppo, per_atleta and bool(gruppo),
                                            None if descr == "—" else descr)
            except Exception as e:
                st.error(str(e))
        ris = ss[KU + "ris"]
        if ris is not None and ris["compito"] == "raggruppamento":
            _risultato_kmeans_utente(ris, gruppo)
        return

    # ---------------------------------------------------------- SUPERVISIONATI
    possibili = ml.bersagli_possibili(dati, candidate, modello)
    if not possibili:
        st.error(f"Nel dataset non c'è una colonna adatta come bersaglio per {modello}. "
                 + ("Serve una colonna sì/no o con poche categorie."
                    if modello == "Regressione Logistica" else
                    "Serve una colonna numerica con più valori diversi."))
        return
    a, b = st.columns(2)
    bersaglio = a.selectbox("Variabile bersaglio (target)", possibili, key=KU + f"b_{modello}")
    info = schema.set_index("colonna").loc[bersaglio]
    descr = " · ".join(str(x) for x in (info.get("significato"), info.get("unità"))
                       if isinstance(x, str) and x.strip())
    serie = dati[bersaglio].dropna()
    b.markdown(f"<br><small>{descr + ' · ' if descr else ''}{serie.nunique()} valori diversi"
               + (f", da {serie.min():g} a {serie.max():g}"
                  if pd.api.types.is_numeric_dtype(serie) else "") + "</small>",
               unsafe_allow_html=True)

    compiti = ml.compiti_ammessi(dati, bersaglio, modello)
    if len(compiti) > 1:
        sugg = ml.compito_suggerito(dati[bersaglio])
        compito = st.radio("Tratta il bersaglio come…", compiti, horizontal=True,
                           index=compiti.index(sugg) if sugg in compiti else 0,
                           format_func=lambda c: {"regressione": "numero (regressione)",
                                                  "classificazione": "categoria (classificazione)"}[c],
                           key=KU + f"comp_{bersaglio}")
    else:
        compito = compiti[0]
    classe_pos = None
    if compito == "classificazione":
        classi = sorted(serie.astype(str).unique())
        if len(classi) == 2:
            noti = [c for c in classi if c.lower() in ("1", "1.0", "sì", "si", "yes", "true")]
            pred = noti[0] if noti else serie.astype(str).value_counts().index[-1]
            classe_pos = st.selectbox("Classe da riconoscere («evento»)", classi,
                                      index=classi.index(pred), key=KU + f"pos_{bersaglio}")
        else:
            st.caption(f"Classificazione con {len(classi)} classi: {', '.join(classi[:10])}"
                       + ("…" if len(classi) > 10 else ""))

    altre = [c for c in candidate if c != bersaglio]
    predittori = st.multiselect("Variabili in ingresso", altre, altre, key=KU + f"x_{bersaglio}")
    avvisi = ml.avvisi_fuga(dati, bersaglio, predittori)
    if avvisi:
        st.warning("⚠️ Possibile fuga di informazione:\n\n" + "\n".join(f"- {a}" for a in avvisi)
                   + "\n\nSe è così, togli quelle colonne dalle variabili in ingresso.")

    opzioni_div = (["per atleta", "casuale"] if gruppo else ["casuale"])
    a, b = st.columns(2)
    divisione = a.radio("Come scegliere le righe di verifica", opzioni_div, horizontal=True,
                        key=KU + "div",
                        help="Per atleta: chi sta nella verifica non è mai stato visto in "
                             "addestramento. Casuale: righe a caso.")
    if not gruppo:
        b.caption("Nessuna colonna «identificativo atleta»: divisione casuale. Se il file ha più "
                  "righe per persona, indica la colonna al passo 2.")
    par, soglia = {}, 0.5
    with st.expander("Impostazioni (facoltative)"):
        quota = st.slider("Quota di dati tenuta da parte per la verifica", 0.15, 0.4, 0.25, 0.05,
                          key=KU + "quota")
        if modello in ("Random Forest", "XGBoost"):
            par["alberi"] = st.slider("Numero di alberi", 50, 600, 300, 50, key=KU + "alb")
            prof = st.slider("Profondità massima (0 = libera)", 0, 12,
                             0 if modello == "Random Forest" else 3, key=KU + f"pr_{modello}")
            par["profondita"] = prof or None
        if modello == "XGBoost":
            par["ritmo"] = st.slider("Velocità di apprendimento", 0.01, 0.3, 0.05, 0.01,
                                     key=KU + "lr")
    if compito == "classificazione" and classe_pos is not None:
        soglia = st.slider("Soglia di allarme", 0.05, 0.95, 0.5, 0.05, key=KU + "soglia")

    a, b = st.columns(2)
    if a.button(f"▶️ Esegui {modello}", type="primary", key=KU + "run", **LARGO):
        try:
            with st.spinner("Addestramento…"):
                ss[KU + "ris"] = ml.addestra_generico(dati, modello, bersaglio, compito, predittori,
                                           categoriali, gruppo, divisione, quota, 7, soglia, par,
                                           classe_pos)
        except Exception as e:
            ss[KU + "ris"] = None
            st.error(f"Analisi non riuscita: {e}")
    if b.button("⚖️ Confronta tutti i modelli su questo bersaglio", key=KU + "cmp", **LARGO):
        try:
            with st.spinner("Un addestramento per modello…"):
                ss[KU + "confronto"] = (bersaglio, ml.confronta_tutti_generico(
                    dati, bersaglio, compito, predittori, categoriali, gruppo, divisione, quota))
        except Exception as e:
            st.error(f"Confronto non riuscito: {e}")

    ris = ss[KU + "ris"]
    if ris is not None and ris["compito"] != "raggruppamento":
        st.divider()
        _risultato(ris, schema)

    cmp = ss[KU + "confronto"]
    if cmp is not None:
        b_cmp, tab = cmp
        st.divider()
        st.subheader(f"Confronto fra modelli · bersaglio «{b_cmp}»")
        st.dataframe(tab, hide_index=True, **LARGO)
        mostra(ml.g_confronto(tab))
        validi = tab[~tab["modello"].str.contains("riferimento")]
        top = validi.loc[validi["R² verifica"].idxmax()]
        lettura([f"R² in verifica più alto: <b>{top['modello']}</b> (R² = {top['R² verifica']:.2f}, "
                 f"MAE = {top['MAE verifica']:.2f}).",
                 "Se la differenza con il modello lineare è piccola, preferisci il più semplice."])
        scarica_json(tab, "confronto_modelli.json", "⬇️ Scarica il confronto (JSON)", KU + "dl_cmp")


def _risultato(ris: dict, schema: pd.DataFrame) -> None:
    unita = schema.set_index("colonna").get("unità", pd.Series(dtype=str)).get(ris["bersaglio"], "")
    unita = unita if isinstance(unita, str) else ""
    st.subheader(f"Risultato · {ris['modello']} su «{ris['bersaglio']}»")
    for a in ris["avvisi"]:
        st.info(a)
    m, rif = ris["misure"], ris["riferimento"]
    c = st.columns(4)
    c[0].metric("R² (verifica)", f"{m['R² (verifica)']:.3f}",
                delta=f"{m['R² (verifica)'] - rif['R² (verifica)']:+.3f} sul riferimento")
    c[1].metric("MAE (verifica)" + (f" · {unita}" if unita and ris["compito"] == "regressione" else ""),
                f"{m['MAE (verifica)']:.3f}",
                delta=f"{m['MAE (verifica)'] - rif['MAE (verifica)']:+.3f} sul riferimento",
                delta_color="inverse")
    c[2].metric("R² (addestramento)", f"{m['R² (addestramento)']:.3f}")
    c[3].metric("MAE (addestramento)", f"{m['MAE (addestramento)']:.3f}")
    if ris["compito"] == "classificazione":
        c2 = st.columns(4)
        c2[0].metric("AUC", "—" if pd.isna(m["AUC"]) else f"{m['AUC']:.3f}")
        c2[1].metric("Accuratezza", f"{m['Accuratezza'] * 100:.0f}%")
        c2[2].metric("Accuratezza bilanciata", f"{m['Accuratezza bilanciata'] * 100:.0f}%")
        if "Sensibilità (eventi riconosciuti)" in m:
            c2[3].metric("Sensibilità", f"{m['Sensibilità (eventi riconosciuti)'] * 100:.0f}%")
    dettaglio = (f"verificato su {ris['n_verifica']} righe di {len(ris['atleti_verifica'])} atleti mai "
                 f"visti" if ris.get("atleti_verifica") else
                 f"verificato su {ris['n_verifica']} righe scelte a caso")
    st.caption(f"Addestrato su {ris['n_addestramento']} righe, {dettaglio}.")
    lettura(ml.commenta_generico(ris) if ris["compito"] == "classificazione" else ml.commenta(ris, unita))

    a, b = st.columns(2)
    with a:
        if ris["compito"] == "regressione":
            mostra(ml.g_osservato_stimato(ris))
        else:
            mostra(ml.g_confusione_multi(ris))
    with b:
        fig = ml.g_importanza(ris)
        if fig is not None:
            mostra(fig)
    if ris["compito"] == "classificazione" and len(ris["classi"]) == 2:
        tmp = dict(ris)
        t = ris["tabella"].copy()
        t["osservato"] = (t["osservato"] == ris["classe_positiva"]).astype(int)
        tmp["tabella"] = t
        a, _ = st.columns(2)
        with a:
            mostra(ml.g_probabilita(tmp))
    if ris.get("coefficienti") is not None:
        with st.expander("I coefficienti del modello lineare"):
            st.dataframe(ris["coefficienti"], hide_index=True, **LARGO)
    with st.expander("Le previsioni, riga per riga"):
        st.dataframe(ris["tabella"], hide_index=True, **LARGO, height=300)

    pacchetto = {
        "file": st.session_state[KU + "nome"], "modello": ris["modello"],
        "bersaglio": ris["bersaglio"], "compito": ris["compito"],
        "classi": ris.get("classi"), "divisione": ris["divisione"],
        "variabili_in_ingresso": ris["predittori"],
        "misure": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in m.items()},
        "riferimento": {k: round(float(v), 4) for k, v in rif.items()},
        "avvisi": ris["avvisi"],
        "importanza": ris["importanza"].to_dict(orient="records") if ris["importanza"] is not None else None,
        "previsioni": json.loads(ris["tabella"].to_json(orient="records", force_ascii=False)),
        "prodotto_il": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.download_button("⬇️ Scarica il risultato completo (JSON)",
                       json.dumps(pacchetto, indent=2, ensure_ascii=False).encode("utf-8"),
                       f"risultato_{ris['modello'].replace(' ', '_')}.json", "application/json",
                       key=KU + "dl_ris")


def _risultato_kmeans_utente(ris: dict, gruppo: str | None) -> None:
    st.divider()
    st.subheader(f"Risultato · K-Means con k = {ris['k']}")
    nomi = list(ris["misure"])
    c = st.columns(2)
    c[0].metric("R² (varianza spiegata dai gruppi)", f"{ris['misure'][nomi[0]]:.3f}")
    c[1].metric("MAE (distanza media dal centro, dev. std.)", f"{ris['misure'][nomi[1]]:.3f}")
    lettura(ml.commenta(ris))
    a, b = st.columns(2)
    with a:
        mostra(ml.g_mappa_gruppi(ris))
    with b:
        mostra(ml.g_gomito(ris))
    st.markdown("**Identikit dei gruppi**")
    st.dataframe(ris["identikit"], hide_index=True, **LARGO)
    if ris["composizione"] is not None:
        st.markdown(f"**Composizione per {ris['categoria']}**")
        st.dataframe(ris["composizione"], **LARGO)
    with st.expander("Il gruppo di ogni riga"):
        st.dataframe(ris["tabella"].drop(columns=["pca_1", "pca_2"]), hide_index=True, **LARGO)
    pacchetto = {"file": st.session_state[KU + "nome"], "modello": "K-Means", "k": ris["k"],
                 "variabili": ris["variabili"],
                 "misure": {k: round(float(v), 4) for k, v in ris["misure"].items()},
                 "identikit": json.loads(ris["identikit"].to_json(orient="records", force_ascii=False)),
                 "assegnazioni": json.loads(ris["tabella"].drop(columns=["pca_1", "pca_2"])
                                            .to_json(orient="records", force_ascii=False))}
    st.download_button("⬇️ Scarica il risultato completo (JSON)",
                       json.dumps(pacchetto, indent=2, ensure_ascii=False).encode("utf-8"),
                       f"kmeans_k{ris['k']}.json", "application/json", key=KU + "dl_km")


# ###########################################################################
# 4 · RL CALCIO E RL NUTRIZIONE
# ###########################################################################
# sezione_rl.py — Sport AI Lab Start · sezioni di reinforcement learning
#
# Un'unica interfaccia per i due ambienti (rl_calcio.py e rl_nutrizione.py):
#
#     1 · Bonus e malus           che cosa premiare e che cosa punire
#     2 · Cosa vede l'agente      quali variabili entrano nello stato
#     3 · Allena e confronta      addestramento, confronto, lettura della politica
#     4 · Episodio e dati         un episodio passo per passo, download JSON
#     📚 Didattica

def rendi_rl(D) -> None:
    ss = st.session_state
    K = f"rl_{D.CHIAVE}_"
    testi = T.RL[D.CHIAVE]

    def carica_preset(nome: str) -> None:
        valori = {r.chiave: r.valore for r in D.RICOMPENSE} | D.PRESET[nome]
        for k, v in valori.items():
            ss[K + "r_" + k] = float(v)

    if K + "r_" + D.RICOMPENSE[0].chiave not in ss:
        carica_preset(next(iter(D.PRESET)))
    for v in D.VARIABILI:
        ss.setdefault(K + "v_" + v.nome, v.nome in D.OSSERVATE_INIZIALI)
    for nome in ("ris", "confronto", "qprec", "importanza", "episodio"):
        ss.setdefault(K + nome, None)
    ss.setdefault(K + "storico", [])

    st.title(D.TITOLO)
    intro(testi["intro"])

    t1, t2, t3, t4, t5 = st.tabs(["1 · Bonus e malus", "2 · Cosa vede l'agente",
                                  "3 · Allena e confronta", "4 · Episodio e dati", "📚 Didattica"])

    # ================================================================ 1. RICOMPENSE
    with t1:
        st.subheader("Che cosa premiare e che cosa punire")
        st.caption("Parti da uno stile già pronto, poi modifica i singoli valori. "
                   "Valori positivi = bonus, negativi = malus.")
        cols = st.columns(len(D.PRESET))
        for col, nome in zip(cols, D.PRESET):
            if col.button(nome, key=K + "p_" + nome, **LARGO):
                carica_preset(nome)
                st.rerun()
        for i in range(0, len(D.RICOMPENSE), 4):
            cols = st.columns(4)
            for col, r in zip(cols, D.RICOMPENSE[i:i + 4]):
                col.number_input(r.etichetta, float(r.minimo), float(r.massimo),
                                 step=float(r.passo), key=K + "r_" + r.chiave,
                                 help=r.aiuto or None)
        st.markdown("**Contesto**")
        cols = st.columns(len(D.CONTESTO))
        for col, c in zip(cols, D.CONTESTO):
            col.slider(c.etichetta, float(c.minimo), float(c.massimo), float(c.valore),
                       float(c.passo), key=K + "c_" + c.chiave, help=c.aiuto or None)

    ric = {r.chiave: float(ss[K + "r_" + r.chiave]) for r in D.RICOMPENSE}
    ctx = {c.chiave: float(ss.get(K + "c_" + c.chiave, c.valore)) for c in D.CONTESTO}

    # ================================================================ 2. STATO
    with t2:
        st.subheader("Quali variabili può vedere l'agente?")
        st.markdown(testi["stato"])
        for v in D.VARIABILI:
            st.checkbox(f"**{v.etichetta}** — {' · '.join(v.livelli)}", key=K + "v_" + v.nome,
                        help=v.descrizione)
            st.caption(v.descrizione)
        osservate = [v.nome for v in D.VARIABILI if ss[K + "v_" + v.nome]]
        n_stati = int(np.prod([len(v.livelli) for v in D.VARIABILI if v.nome in osservate])) \
            if osservate else 1
        c = st.columns(3)
        c[0].metric("Variabili osservate", f"{len(osservate)} su {len(D.VARIABILI)}")
        c[1].metric("Situazioni distinte", n_stati)
        c[2].metric("Celle della Q-table", n_stati * len(D.AZIONI))
        st.caption("Più variabili = decisioni più su misura, ma più situazioni da imparare e quindi "
                   "più episodi di allenamento. Le variabili non spuntate continuano a influenzare "
                   "il gioco: l'agente semplicemente non le vede.")

        with st.expander("Parametri di apprendimento (facoltativi)"):
            a, b, cc, d = st.columns(4)
            a.select_slider("Episodi di allenamento", D.EPISODI_SCELTE,
                            D.IMPOSTAZIONI["episodi"], key=K + "ep")
            b.slider("α minimo (velocità di apprendimento)", 0.005, 0.2, 0.01, 0.005,
                     key=K + "alfa")
            cc.slider("γ peso del futuro", 0.0, 0.99, float(D.IMPOSTAZIONI["gamma"]), 0.01,
                      key=K + "gamma")
            d.slider("ε esplorazione iniziale", 0.0, 1.0, 0.30, 0.05, key=K + "eps")

    imp = rl.Impostazioni(episodi=int(ss.get(K + "ep", D.IMPOSTAZIONI["episodi"])),
                         alfa=float(ss.get(K + "alfa", 0.01)),
                         gamma=float(ss.get(K + "gamma", D.IMPOSTAZIONI["gamma"])),
                         epsilon=float(ss.get(K + "eps", 0.30)))

    # ================================================================ 3. ALLENA
    with t3:
        st.subheader("Allena l'agente e confrontalo")
        st.caption(f"Vede: **{', '.join(osservate) if osservate else 'nulla'}** · "
                   f"{imp.episodi} episodi · bonus e malus: "
                   + " · ".join(f"{r.etichetta.split(' ', 1)[-1]} {ric[r.chiave]:+g}"
                                for r in D.RICOMPENSE))
        if st.button("▶️ Allena l'agente", type="primary", key=K + "train", **LARGO):
            with st.spinner(f"L'agente gioca {imp.episodi} episodi…"):
                ris = rl.allena(D, ric, ctx, imp, osservate)
                ss[K + "ris"] = ris
                ss[K + "confronto"] = rl.confronto(D, ris, ric, ctx)
                ss[K + "qprec"] = rl.precisione_q(D, ris, ric, ctx)
                ss[K + "importanza"] = rl.importanza_variabili(D, ris) if osservate else None
                ss[K + "episodio"] = None
            tab = ss[K + "confronto"]
            ag = tab[tab["strategia"].str.contains("agente")].iloc[0]
            fisse = tab[~tab["strategia"].str.contains("agente")].iloc[0]
            ss[K + "storico"].append({
                "variabili osservate": ", ".join(osservate) or "nessuna",
                "situazioni": ris["n_stati"], "episodi": imp.episodi,
                "ricompensa agente": ag["ricompensa media"],
                "migliore strategia fissa": f"{fisse['strategia']} ({fisse['ricompensa media']})",
                "R² Q": round(ss[K + "qprec"]["R²"], 3), "MAE Q": round(ss[K + "qprec"]["MAE"], 2),
                "ricompense": ", ".join(f"{k} {v:+g}" for k, v in ric.items())})

        ris = ss[K + "ris"]
        if ris is None:
            st.info("Scegli bonus, malus e variabili, poi premi **Allena l'agente**.")
        else:
            if (ris["ricompense"] != ric or ris["contesto"] != ctx
                    or ris["osservate"] != osservate):
                st.warning("Hai cambiato bonus, malus, contesto o variabili dopo l'allenamento: "
                           "riallena per vederne l'effetto.")
            _risultati(D, K, ris, ric, ctx)

    # ================================================================ 4. EPISODIO
    with t4:
        ris = ss[K + "ris"]
        if ris is None:
            st.warning("Prima allena l'agente al passo 3.")
        else:
            st.subheader(f"Un {D.EPISODIO} giocato dall'agente, {D.UNITA} per {D.UNITA}")
            seme = st.number_input("Episodio n°", 1, 9999, 1, key=K + "seme")
            if st.button("▶️ Simula", type="primary", key=K + "sim", **LARGO) \
                    or ss[K + "episodio"] is None:
                ss[K + "episodio"] = rl.gioca(D, rl.scelta_agente(D, ris), ris["ricompense"],
                                             ris["contesto"], int(seme))
            log = ss[K + "episodio"]
            m = D.metriche_episodio(log)
            cols = st.columns(len(m))
            for col, (k, v) in zip(cols, m.items()):
                col.metric(k, v)
            mostra(D.g_episodio(log))
            st.dataframe(log, hide_index=True, **LARGO, height=380)

            st.divider()
            st.subheader("Scarica i dati (JSON)")
            a, b = st.columns(2)
            with a:
                scarica_json(log, f"rl_{D.CHIAVE}_episodio_{int(seme)}.json",
                             f"⬇️ {D.EPISODIO.capitalize()} passo per passo", K + "dl_e")
                scarica_json(ris["storia"], f"rl_{D.CHIAVE}_storia_allenamento.json",
                             "⬇️ Storia dell'allenamento (un episodio per riga)", K + "dl_s")
                scarica_json(ss[K + "confronto"], f"rl_{D.CHIAVE}_confronto.json",
                             "⬇️ Confronto con le strategie fisse", K + "dl_c")
                if ss[K + "storico"]:
                    scarica_json(pd.DataFrame(ss[K + "storico"]),
                                 f"rl_{D.CHIAVE}_storico_addestramenti.json",
                                 "⬇️ Storico degli addestramenti", K + "dl_h")
            with b:
                scarica_json(rl.tabella_politica(D, ris), f"rl_{D.CHIAVE}_politica.json",
                             "⬇️ Decisione appresa per ogni situazione", K + "dl_p")
                scarica_json(rl.tabella_q(D, ris), f"rl_{D.CHIAVE}_q_table.json",
                             "⬇️ Q-table", K + "dl_q")
                if ss[K + "importanza"] is not None:
                    scarica_json(ss[K + "importanza"], f"rl_{D.CHIAVE}_importanza.json",
                                 "⬇️ Importanza delle variabili", K + "dl_i")
                cfg = {"variabili_osservate": ris["osservate"], "ricompense": ris["ricompense"],
                       "contesto": ris["contesto"], "impostazioni": ris["impostazioni"],
                       "precisione_valori_Q": ss[K + "qprec"]}
                st.download_button("⬇️ Configurazione usata",
                                   json.dumps(cfg, indent=2, ensure_ascii=False).encode("utf-8"),
                                   f"rl_{D.CHIAVE}_configurazione.json", "application/json",
                                   key=K + "dl_cfg")

    # ================================================================ DIDATTICA
    with t5:
        s1, s2, s3 = st.tabs(["Come funziona", "Esercizi", "Glossario"])
        with s1:
            st.markdown(testi["intro_lunga"])
            st.markdown("#### Le variabili dello stato")
            st.dataframe(pd.DataFrame([{"variabile": v.etichetta, "livelli": " · ".join(v.livelli),
                                        "perché conta": v.descrizione} for v in D.VARIABILI]),
                         hide_index=True, **LARGO)
            st.markdown(testi["azioni"])
            st.markdown(T.RL_COMUNE)
        with s2:
            for i, (tit, cosa, consegna) in enumerate(testi["esercizi"], 1):
                scheda(f"{i}. {tit}", [f"<b>Che cosa fare</b>: {cosa}",
                                       f"<b>Consegna</b>: {consegna}"])
        with s3:
            for termine, definizione in T.GLOSSARIO_RL:
                st.markdown(f"**{termine}** — {definizione}")


# =========================================================================
# RISULTATI DELL'ADDESTRAMENTO
# =========================================================================

def _risultati(D, K: str, ris: dict, ric: dict, ctx: dict) -> None:
    ss = st.session_state
    tab = ss[K + "confronto"]
    q = ss[K + "qprec"]
    ag = tab[tab["strategia"].str.contains("agente")].iloc[0]
    fisse = tab[~tab["strategia"].str.contains("agente")]
    top = fisse.iloc[0]

    c = st.columns(4)
    c[0].metric("Ricompensa media dell'agente", f"{ag['ricompensa media']:.1f}",
                delta=f"{ag['ricompensa media'] - top['ricompensa media']:+.1f} sulla migliore fissa")
    c[1].metric("Situazioni imparate", f"{int((ris['visite'] >= rl.MIN_VISITE).sum())} su {ris['n_stati']}")
    c[2].metric("R² dei valori Q", f"{q['R²']:.3f}")
    c[3].metric("MAE dei valori Q", f"{q['MAE']:.2f}")
    st.caption("R² e MAE dei valori Q: quanto la ricompensa futura prevista dall'agente è vicina a "
               "quella ottenuta davvero — le stesse misure dei modelli di machine learning.")

    a, b = st.columns(2)
    with a:
        mostra(rl.g_apprendimento(D, ris))
    with b:
        mostra(rl.g_confronto(tab))

    righe = []
    if ag["ricompensa media"] >= top["ricompensa media"]:
        righe.append(f"L'agente ottiene <b>{ag['ricompensa media']:.1f}</b> punti per episodio, più "
                     f"della migliore strategia fissa (<i>{top['strategia']}</i>, "
                     f"{top['ricompensa media']:.1f}): ha imparato ad <b>adattare</b> la decisione "
                     "alla situazione.")
    else:
        righe.append(f"La strategia <i>{top['strategia']}</i> fa meglio dell'agente "
                     f"({top['ricompensa media']:.1f} contro {ag['ricompensa media']:.1f}). Con "
                     "tante variabili servono più episodi: aumentali, oppure togli una variabile "
                     "poco utile.")
    righe.append("Per capire se una variabile serve, togli la spunta, riallena e confronta le "
                 "righe dello <b>storico degli addestramenti</b> qui sotto.")
    lettura(righe)

    st.markdown("**Confronto su 200 episodi identici per tutti**")
    st.dataframe(tab.rename(columns=getattr(D, "RINOMINA_CONFRONTO", {})), hide_index=True, **LARGO)

    if ris["osservate"]:
        st.divider()
        st.subheader("Che cosa ha imparato")
        imp_v = ss[K + "importanza"]
        a, b = st.columns(2)
        with a:
            mostra(rl.g_importanza(imp_v))
            st.caption("Per ogni situazione si cambia una sola variabile e si guarda se la decisione "
                       "migliore cambia in modo rilevante (almeno 1 punto di ricompensa attesa).")
        with b:
            nomi = {v.etichetta: v.nome for v in D.VARIABILI if v.nome in ris["osservate"]}
            scelta = st.selectbox("Come cambia la decisione al variare di…", list(nomi),
                                  key=K + "uso_var")
            st.dataframe(rl.uso_per_variabile(D, ris, ric, ctx, nomi[scelta]), **LARGO)
            st.caption("Percentuale di volte in cui l'agente sceglie ciascuna azione, per livello "
                       "della variabile, giocando 120 episodi.")

        st.markdown("**Mappa delle decisioni**")
        oss = [v for v in D.VARIABILI if v.nome in ris["osservate"]]
        if len(oss) >= 2:
            c1, c2 = st.columns(2)
            vr = c1.selectbox("Righe", [v.etichetta for v in oss], index=0, key=K + "map_r")
            altre = [v.etichetta for v in oss if v.etichetta != vr]
            vc = c2.selectbox("Colonne", altre, index=min(1, len(altre) - 1) if len(altre) > 1 else 0,
                              key=K + "map_c")
            restanti = [v for v in oss if v.etichetta not in (vr, vc)]
            fissi = {}
            if restanti:
                st.caption("Le altre variabili sono tenute fisse a:")
                cols = st.columns(len(restanti))
                for col, v in zip(cols, restanti):
                    # di partenza, il livello più frequente: la mappa è subito leggibile
                    i = D.VARIABILI.index(v)
                    asse = tuple(j for j in range(ris["visite"].ndim) if j != i)
                    piu_vista = int(np.argmax(ris["visite"].sum(axis=asse)))
                    liv = col.selectbox(v.etichetta, v.livelli, index=piu_vista,
                                        key=K + "fix_" + v.nome)
                    fissi[v.nome] = v.livelli.index(liv)
            nome_r = next(v.nome for v in oss if v.etichetta == vr)
            nome_c = next(v.nome for v in oss if v.etichetta == vc)
            a, _ = st.columns([3, 1])
            with a:
                mostra(rl.g_mappa(D, ris, nome_r, nome_c, fissi))
        with st.expander("Tutte le situazioni imparate, in tabella"):
            st.dataframe(rl.tabella_politica(D, ris), hide_index=True, **LARGO, height=320)

    st.divider()
    st.subheader("Storico degli addestramenti")
    st.dataframe(pd.DataFrame(ss[K + "storico"]), hide_index=True, **LARGO)
    st.caption("Ogni volta che premi «Allena» si aggiunge una riga: confronta agenti che vedono "
               "variabili diverse o che ricevono bonus e malus diversi.")
    if st.button("🗑️ Svuota lo storico", key=K + "svuota"):
        ss[K + "storico"] = []
        st.rerun()


# ###########################################################################
# 5 · HOME E NAVIGAZIONE
# ###########################################################################




VERSIONE = "1.0"
st.markdown(CSS, unsafe_allow_html=True)

PAGINE = ["🏠 Home", "🥅 KPI Sport", "🥗 KPI Nutrizione", "📂 I tuoi dati", "⚽ RL Calcio",
          "🍝 RL Nutrizione"]
st.session_state.setdefault("pagina", PAGINE[0])


def vai(pagina: str) -> None:
    st.session_state["pagina"] = pagina


with st.sidebar:
    st.title("🧪 Sport AI Lab")
    st.caption(f"Edizione Start · v{VERSIONE}")
    st.radio("Sezione", PAGINE, key="pagina")
    st.divider()
    pag = st.session_state["pagina"]
    if pag == "📂 I tuoi dati":
        st.markdown("**Percorso**\n\n1. Carica il dataset\n2. Analizza e pulisci\n"
                    "3. Modelli ML\n\n📚 Didattica: come preparare il file")
    elif pag in ("🥅 KPI Sport", "🥗 KPI Nutrizione"):
        st.markdown("**Percorso**\n\n1. Genera il dataset\n2. Analizza e pulisci\n"
                    "3. Modelli ML\n\n📚 Didattica: tutte le spiegazioni")
    elif pag in ("⚽ RL Calcio", "🍝 RL Nutrizione"):
        st.markdown("**Percorso**\n\n1. Bonus e malus\n2. Cosa vede l'agente\n"
                    "3. Allena e confronta\n4. Episodio e dati\n\n"
                    "📚 Didattica: come impara l'agente")
    if not ml.XGB_OK:
        st.caption("ℹ️ `xgboost` non installato: XGBoost usa il Gradient Boosting di "
                   "scikit-learn.")
    st.divider()
    st.caption("Dati interamente simulati, a scopo didattico.")


def home() -> None:
    st.title("🧪 Sport AI Lab · Start")
    st.caption("Laboratorio di machine learning per chinesiologi e nutrizionisti")
    intro("Cinque sezioni, un metodo semplice: <b>genera i dati, capiscili, puliscili, poi "
          "usa i modelli</b> e giudicali con due numeri — R² e MAE. Ogni tabella si scarica "
          "in JSON.")

    st.markdown("#### Machine learning sui KPI")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("""
<div class="porta"><div class="occhiello">Sezione 1</div><h3>🥅 KPI Sport</h3>
<p>Una squadra di calcio per una stagione: GPS, RPE, sonno, HRV, prestazione, infortuni.</p>
<ul><li>prevedi l'RPE dal carico esterno</li><li>stima il rischio di infortunio</li>
<li>trova i profili atletici della rosa</li></ul></div>""", unsafe_allow_html=True)
        st.button("Entra in KPI Sport", type="primary", on_click=vai, args=("🥅 KPI Sport",),
                  key="h1", **LARGO)
    with b:
        st.markdown("""
<div class="porta"><div class="occhiello">Sezione 2</div><h3>🥗 KPI Nutrizione</h3>
<p>Atleti seguiti giorno per giorno: diario alimentare, CGM, dispendio, recupero.</p>
<ul><li>prevedi il picco glicemico dalla colazione</li><li>allerta la bassa disponibilità
energetica</li><li>raggruppa gli atleti per abitudini</li></ul></div>""", unsafe_allow_html=True)
        st.button("Entra in KPI Nutrizione", type="primary", on_click=vai,
                  args=("🥗 KPI Nutrizione",), key="h2", **LARGO)

    st.markdown("""
<div class="porta" style="min-height:auto"><div class="occhiello">Sezione 3</div>
<h3>📂 I tuoi dati</h3>
<p>Carica un tuo dataset (CSV, Excel o JSON) con colonne qualsiasi: l'app riconosce tipi e
ruoli, tu li correggi, lo pulisci e lo analizzi con gli stessi cinque modelli, che si adattano
alle tue colonne — anche con bersagli a più di due classi.</p></div>""", unsafe_allow_html=True)
    st.button("Entra in I tuoi dati", type="primary", on_click=vai, args=("📂 I tuoi dati",),
              key="h5", **LARGO)

    st.markdown("#### Reinforcement learning: un agente che impara per tentativi")
    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("""
<div class="porta"><div class="occhiello">Sezione 4</div><h3>⚽ RL Calcio</h3>
<p>Un allenatore virtuale sceglie la tattica ogni 5 minuti: pressing alto, blocco medio,
difesa bassa, contropiede, possesso.</p>
<ul><li>bonus e malus decisi da te (gol fatto, gol subito, espulsione…)</li>
<li>stato: punteggio, energia, minuto, pressing avversario, gol appena subito, uomini in campo</li>
<li>scegli tu quali variabili l'agente può vedere</li></ul></div>""", unsafe_allow_html=True)
        st.button("Entra in RL Calcio", type="primary", on_click=vai,
                  args=("⚽ RL Calcio",), key="h3", **LARGO)
    with d:
        st.markdown("""
<div class="porta"><div class="occhiello">Sezione 5</div><h3>🍝 RL Nutrizione</h3>
<p>Un nutrizionista virtuale sceglie ogni giorno i carboidrati (3–11 g/kg) per un mese di
preparazione.</p>
<ul><li>bonus e malus decisi da te (gara riuscita, bassa disponibilità energetica, disturbi GI…)</li>
<li>stato: tipo di giornata, gara domani, glicogeno, bilancio energetico, sonno</li>
<li>riscopre da solo le linee guida?</li></ul></div>""", unsafe_allow_html=True)
        st.button("Entra in RL Nutrizione", type="primary", on_click=vai,
                  args=("🍝 RL Nutrizione",), key="h4", **LARGO)

    st.divider()
    st.subheader("I cinque modelli")
    for nome in ml.MODELLI:
        st.markdown(f"**{nome}** — {T.COSA_FA[nome]}")

    with st.expander("Avvertenza sui dati e sull'uso"):
        st.markdown("I dati sono **interamente simulati**: gli ordini di grandezza sono "
                    "plausibili, ma non provengono da persone reali. I risultati servono a "
                    "imparare il metodo e **non sono trasferibili a decisioni su atleti veri**, "
                    "che richiedono validazione su dati propri e tutela dei dati sanitari (GDPR).")
    st.caption("Sport AI Lab Start · licenza MIT · © 2026 Christian Fiorino")


pag = st.session_state["pagina"]
if pag == "🥅 KPI Sport":
    rendi_kpi("sport")
elif pag == "🥗 KPI Nutrizione":
    rendi_kpi("nutrizione")
elif pag == "📂 I tuoi dati":
    rendi_utente()
elif pag == "⚽ RL Calcio":
    rendi_rl(rl.CALCIO)
elif pag == "🍝 RL Nutrizione":
    rendi_rl(rl.NUTRIZIONE)
else:
    home()
