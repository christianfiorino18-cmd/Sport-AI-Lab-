"""
ml.py — Sport AI Lab Start · dati e machine learning

Tutto ciò che serve alle sezioni KPI e «I tuoi dati», senza Streamlit
(si usa anche da notebook):

    1. DATI SIMULATI         catalogo delle colonne, generazione, errori, pulizia
    2. DATI DELLO STUDENTE   lettura CSV/Excel/JSON, tipi e ruoli, pulizia
    3. I CINQUE MODELLI      addestramento, R² e MAE, K-Means, grafici
    4. MODELLI GENERICI      gli stessi modelli su colonne qualsiasi, anche multiclasse

Le funzioni della parte 2 e 4 che hanno lo stesso nome di quelle della 1 e 3
portano il suffisso _utente o _generico (es. pulisci_utente, addestra_generico).

Copyright (C) 2026  Christian Fiorino — licenza MIT
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402


# ###########################################################################
# 1 · DATI SIMULATI
# ###########################################################################
# dati.py — Sport AI Lab Start · generazione e pulizia dei dataset
#
# Due dataset simulati, uno per sezione:
#
#     sport       una squadra di calcio, una riga per giocatore per settimana
#     nutrizione  un gruppo di atleti, una riga per atleta per giorno
#
# Per ciascuno c'è un CATALOGO che dice cosa significa ogni colonna, in che
# unità è espressa, qual è il suo ruolo (identificativo, categoria, misura,
# bersaglio) e quali sono i limiti fisiologici oltre i quali un valore è un
# errore e non un dato raro.
#
# Il modulo non importa Streamlit: si può usare anche da un notebook.

# =========================================================================
# CATALOGO DELLE COLONNE
# =========================================================================

@dataclass
class Colonna:
    descrizione: str
    unita: str
    ruolo: str                 # id · categoria · misura · bersaglio numerico · bersaglio sì/no · trappola
    minimo: float | None = None
    massimo: float | None = None


CATALOGO_SPORT: Dict[str, Colonna] = {
    "id_giocatore": Colonna("Codice del giocatore", "—", "id"),
    "ruolo": Colonna("Ruolo in campo", "—", "categoria"),
    "settimana": Colonna("Settimana della stagione", "n°", "id"),
    "eta": Colonna("Età del giocatore", "anni", "misura", 15, 45),
    "distanza_km": Colonna("Distanza totale percorsa nella settimana (allenamenti + partita), dal GPS",
                           "km", "misura", 5, 60),
    "distanza_alta_int_m": Colonna("Metri percorsi sopra i 19,8 km/h, dal GPS", "m", "misura", 0, 8000),
    "sprint_n": Colonna("Numero di sprint (sopra i 25 km/h) nella settimana", "n°", "misura", 0, 150),
    "accelerazioni_n": Colonna("Accelerazioni sopra i 3 m/s² nella settimana", "n°", "misura", 0, 600),
    "vel_max_kmh": Colonna("Velocità massima raggiunta nella settimana", "km/h", "misura", 15, 38),
    "minuti_partita": Colonna("Minuti giocati in partita nella settimana (fino a 2 partite)",
                              "min", "misura", 0, 200),
    "acwr": Colonna("Acute:Chronic Workload Ratio: distanza della settimana / media delle 4 precedenti. "
                    "Sopra 1,5 il carico sale troppo in fretta", "rapporto", "misura", 0.2, 3.0),
    "sonno_h": Colonna("Ore di sonno medie per notte (questionario)", "h", "misura", 3, 12),
    "hrv_ms": Colonna("Variabilità della frequenza cardiaca a riposo (RMSSD): più alta = più recuperato",
                      "ms", "misura", 10, 200),
    "benessere": Colonna("Questionario di benessere (umore, dolori, stress): 1 pessimo, 10 ottimo",
                         "1–10", "misura", 1, 10),
    "rpe_medio": Colonna("Fatica percepita media delle sedute, scala di Borg CR-10", "0–10",
                         "bersaglio numerico", 0, 10),
    "carico_interno_ua": Colonna("Carico interno = RPE × minuti totali. Contiene già l'RPE",
                                 "UA", "misura", 0, 8000),
    "indice_prestazione": Colonna("Valutazione della prestazione in partita data dallo staff",
                                  "0–100", "bersaglio numerico", 0, 100),
    "infortunio": Colonna("1 = infortunio nelle due settimane successive, 0 = nessuno",
                          "sì/no", "bersaglio sì/no"),
    "giorni_stop": Colonna("Giorni di stop per infortunio. È noto solo DOPO l'infortunio",
                           "giorni", "trappola", 0, 120),
}

CATALOGO_NUTRIZIONE: Dict[str, Colonna] = {
    "id_atleta": Colonna("Codice dell'atleta", "—", "id"),
    "sport": Colonna("Disciplina praticata", "—", "categoria"),
    "sesso": Colonna("Sesso", "F/M", "categoria"),
    "giorno": Colonna("Giorno di monitoraggio", "n°", "id"),
    "tipo_giornata": Colonna("Riposo, leggero, intenso o gara", "—", "categoria"),
    "eta": Colonna("Età dell'atleta", "anni", "misura", 15, 50),
    "massa_kg": Colonna("Massa corporea", "kg", "misura", 35, 130),
    "massa_magra_kg": Colonna("Massa magra (FFM), da plicometria o bioimpedenza", "kg", "misura", 25, 110),
    "energia_kcal": Colonna("Energia introdotta nella giornata, dal diario alimentare", "kcal",
                            "misura", 800, 6500),
    "cho_g": Colonna("Carboidrati totali della giornata", "g", "misura", 50, 1000),
    "pro_g": Colonna("Proteine totali della giornata", "g", "misura", 20, 350),
    "grassi_g": Colonna("Grassi totali della giornata", "g", "misura", 10, 250),
    "fibra_g": Colonna("Fibra alimentare della giornata", "g", "misura", 3, 80),
    "cho_g_kg": Colonna("Carboidrati per kg di massa corporea", "g/kg", "misura", 1, 14),
    "pro_g_kg": Colonna("Proteine per kg di massa corporea", "g/kg", "misura", 0.4, 4),
    "cho_colazione_g": Colonna("Carboidrati della colazione", "g", "misura", 0, 250),
    "ig_colazione": Colonna("Indice glicemico medio della colazione", "0–100", "misura", 20, 100),
    "fibra_colazione_g": Colonna("Fibra della colazione", "g", "misura", 0, 30),
    "dispendio_esercizio_kcal": Colonna("Energia spesa con l'allenamento, da cardiofrequenzimetro",
                                        "kcal", "misura", 0, 3000),
    "variazione_peso_7g_kg": Colonna("Variazione di peso rispetto a 7 giorni prima (bilancia del "
                                     "mattino): negativa se l'atleta sta perdendo peso", "kg",
                                     "misura", -6, 6),
    "disponibilita_energetica": Colonna("Energy Availability = (energia − dispendio) / massa magra. "
                                        "Sotto 30 è bassa", "kcal/kg FFM", "misura", -20, 90),
    "sonno_h": Colonna("Ore di sonno della notte precedente", "h", "misura", 3, 12),
    "idratazione_l": Colonna("Liquidi assunti nella giornata", "L", "misura", 0.5, 8),
    "picco_glicemico": Colonna("Glicemia massima nelle 2 ore dopo colazione, da sensore CGM",
                               "mg/dl", "bersaglio numerico", 60, 260),
    "recupero": Colonna("Punteggio di recupero del mattino dopo (HRV + questionario)", "0–100",
                        "bersaglio numerico", 0, 100),
    "bassa_ea": Colonna("1 = giornata con disponibilità energetica sotto 30 kcal/kg FFM",
                        "sì/no", "bersaglio sì/no"),
}

CATALOGHI = {"sport": CATALOGO_SPORT, "nutrizione": CATALOGO_NUTRIZIONE}
ID_GRUPPO = {"sport": "id_giocatore", "nutrizione": "id_atleta"}

# Colonne che "tradiscono" un bersaglio: contengono la risposta, o sono note
# solo dopo l'evento. Vengono escluse in automatico quando si sceglie quel bersaglio.
ESCLUSE_PER_BERSAGLIO: Dict[str, Dict[str, str]] = {
    "rpe_medio": {"carico_interno_ua": "è calcolato moltiplicando proprio l'RPE per i minuti"},
    "infortunio": {"giorni_stop": "è diverso da zero solo se l'infortunio è già avvenuto"},
    "indice_prestazione": {},
    "picco_glicemico": {},
    "recupero": {},
    # bassa_ea si CALCOLA dal diario completo: se lo dessimo al modello, il modello
    # ricostruirebbe la formula. La domanda utile è un'altra: si può prevedere la
    # giornata critica prima di avere il diario, da colazione, allenamento e profilo?
    "bassa_ea": {"disponibilita_energetica": "bassa_ea è definita come disponibilità energetica < 30",
                 "energia_kcal": "è il numeratore della formula della disponibilità energetica",
                 "cho_g": "carboidrati + proteine + grassi ricostruiscono l'energia della giornata",
                 "pro_g": "carboidrati + proteine + grassi ricostruiscono l'energia della giornata",
                 "grassi_g": "carboidrati + proteine + grassi ricostruiscono l'energia della giornata",
                 "cho_g_kg": "è noto solo a diario completo, a fine giornata",
                 "pro_g_kg": "è noto solo a diario completo, a fine giornata",
                 "fibra_g": "è noto solo a diario completo, a fine giornata",
                 },
}


def catalogo(dominio: str) -> Dict[str, Colonna]:
    return CATALOGHI[dominio]


def tabella_significato(dominio: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """La scheda delle colonne: significato, unità, ruolo e limiti plausibili."""
    righe = []
    for nome, c in CATALOGHI[dominio].items():
        if df is not None and nome not in df.columns:
            continue
        limiti = ("—" if c.minimo is None else f"{c.minimo:g} – {c.massimo:g}")
        righe.append({"colonna": nome, "significato": c.descrizione, "unità": c.unita,
                      "ruolo": c.ruolo, "valori plausibili": limiti})
    return pd.DataFrame(righe)


def bersagli(dominio: str, tipo: str) -> List[str]:
    """tipo = 'numerico' oppure 'binario'."""
    ruolo = "bersaglio numerico" if tipo == "numerico" else "bersaglio sì/no"
    return [k for k, c in CATALOGHI[dominio].items() if c.ruolo == ruolo]


def predittori(dominio: str, df: pd.DataFrame, bersaglio: str | None) -> List[str]:
    """Colonne utilizzabili come ingresso per un bersaglio: niente id, niente
    altri bersagli, niente trappole, niente colonne che contengono la risposta."""
    cat = CATALOGHI[dominio]
    escluse = set(ESCLUSE_PER_BERSAGLIO.get(bersaglio, {})) if bersaglio else set()
    return [c for c in df.columns
            if c in cat and c != bersaglio and c not in escluse
            and cat[c].ruolo in ("misura", "categoria")]


# =========================================================================
# GENERAZIONE · SPORT
# =========================================================================

RUOLI = {   # distanza km/sett, alta intensità m, sprint, accelerazioni
    "Difensore": (24.0, 1700, 16, 210),
    "Centrocampista": (28.5, 2400, 20, 260),
    "Esterno": (27.0, 3000, 34, 280),
    "Attaccante": (24.5, 2300, 30, 240),
}


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def genera_sport(giocatori: int = 22, settimane: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    righe = []
    ruoli = list(RUOLI)
    for g in range(giocatori):
        ruolo = ruoli[g % len(ruoli)]
        d0, hi0, sp0, ac0 = RUOLI[ruolo]
        eta = int(rng.integers(18, 35))
        forma = rng.normal(1.0, 0.08)                  # capacità aerobica individuale
        fragilita = rng.normal(0.0, 0.3)               # predisposizione agli infortuni
        hrv_base = rng.normal(70, 10)
        titolare = rng.random() < 0.65
        storico = [d0 * forma] * 4
        fatica = 0.2
        for s in range(1, settimane + 1):
            doppia = rng.random() < 0.25                    # settimana con due partite
            onda = 1.0 + 0.12 * np.sin(s / 2.3) + rng.normal(0, 0.10)
            picco = rng.random() < 0.08                      # settimana di carico improvviso
            fattore = onda * (1.18 if doppia else 1.0) * (1.35 if picco else 1.0)
            minuti = 0.0
            for _ in range(2 if doppia else 1):
                minuti += (rng.normal(82, 10) if titolare else rng.normal(25, 15))
            minuti = float(np.clip(minuti, 0, 190))
            distanza = d0 * forma * fattore * rng.normal(1, 0.05)
            alta = hi0 * fattore * forma ** 2 * rng.normal(1, 0.10)
            sprint = max(0, rng.poisson(sp0 * fattore))
            acc = max(0, rng.poisson(ac0 * fattore))
            vmax = 30.5 + 1.5 * (ruolo in ("Esterno", "Attaccante")) + 2.0 * (forma - 1) \
                - 0.08 * (eta - 25) + rng.normal(0, 0.8)
            acwr = distanza / np.mean(storico[-4:])
            storico.append(distanza)

            fatica = float(np.clip(0.55 * fatica + 0.9 * (fattore - 0.9) + rng.normal(0, 0.06), 0, 1))
            sonno = float(np.clip(7.7 - 1.2 * fatica + rng.normal(0, 0.55), 4.5, 10))
            hrv = float(np.clip(hrv_base - 18 * fatica + 2.5 * (sonno - 7.5) + rng.normal(0, 5), 20, 140))
            benessere = float(np.clip(7.2 - 2.5 * fatica + 0.6 * (sonno - 7.5) + rng.normal(0, 0.7), 1, 10))

            durata = rng.normal(380, 40)
            rpe = float(np.clip(1.0 + 0.12 * distanza + 0.0006 * alta + 2.2 * fatica
                                - 3.0 * (forma - 1) + rng.normal(0, 0.45), 1, 10))
            carico = rpe * (durata + minuti)

            prestazione = float(np.clip(
                55 + 60 * (forma - 1) + 3.0 * (benessere - 6) - 14 * fatica
                + 0.04 * (hrv - 70) + 0.08 * (minuti - 60) / 2 + rng.normal(0, 5), 0, 100))

            logit = (-3.1 + 4.5 * max(0.0, acwr - 1.2) + 2.0 * fatica
                     + 0.6 * (7.0 - sonno) + 0.05 * (eta - 26) + fragilita
                     + 1.0 * (acwr < 0.8))
            infortunio = int(rng.random() < _sigmoide(logit))
            stop = int(rng.integers(4, 40)) if infortunio else 0

            righe.append({
                "id_giocatore": f"G{g + 1:02d}", "ruolo": ruolo, "settimana": s, "eta": eta,
                "distanza_km": round(distanza, 2), "distanza_alta_int_m": round(alta),
                "sprint_n": int(sprint), "accelerazioni_n": int(acc),
                "vel_max_kmh": round(vmax, 1), "minuti_partita": round(minuti),
                "acwr": round(acwr, 2), "sonno_h": round(sonno, 1), "hrv_ms": round(hrv, 1),
                "benessere": round(benessere, 1), "rpe_medio": round(rpe, 1),
                "carico_interno_ua": round(carico), "indice_prestazione": round(prestazione, 1),
                "infortunio": infortunio, "giorni_stop": stop,
            })
    return pd.DataFrame(righe)


# =========================================================================
# GENERAZIONE · NUTRIZIONE
# =========================================================================

SPORT_NUTRIZIONE = ["Calcio", "Corsa", "Tennis", "Nuoto", "Ciclismo"]
CHO_OBIETTIVO = {"riposo": 3.5, "leggero": 5.0, "intenso": 7.0, "gara": 8.5}
DISPENDIO = {"riposo": 60.0, "leggero": 450.0, "intenso": 900.0, "gara": 1150.0}


def genera_nutrizione(atleti: int = 20, giorni: int = 28, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    righe = []
    for a in range(atleti):
        sport = SPORT_NUTRIZIONE[a % len(SPORT_NUTRIZIONE)]
        sesso = "F" if rng.random() < 0.45 else "M"
        eta = int(rng.integers(18, 36))
        massa = rng.normal(61, 6) if sesso == "F" else rng.normal(73, 8)
        grasso = (rng.normal(21, 3) if sesso == "F" else rng.normal(12, 3)) / 100
        magra = massa * (1 - grasso)
        restrizione = float(np.clip(rng.normal(0.13, 0.10), -0.05, 0.38))   # tende a mangiare poco
        sensibilita = rng.normal(1.0, 0.18)                                # risposta glicemica
        ig_preferito = rng.normal(62, 10)
        sonno_base = rng.normal(7.5, 0.5)
        for gno in range(1, giorni + 1):
            if gno % 7 == 0:
                tipo = "gara" if rng.random() < 0.6 else "riposo"
            elif gno % 7 == 1:
                tipo = "riposo"
            else:
                tipo = "intenso" if rng.random() < 0.5 else "leggero"
            dispendio = max(0.0, DISPENDIO[tipo] * massa / 70 * rng.normal(1, 0.12))
            fabbisogno = 24 * massa * 1.45 + dispendio
            # nei giorni duri si tende a non mangiare abbastanza (sotto-rifornimento)
            duro = 0.07 if tipo in ("intenso", "gara") else 0.0
            energia_obiettivo = fabbisogno * (1 - restrizione - duro + rng.normal(0, 0.07))
            cho_kg = max(1.5, CHO_OBIETTIVO[tipo] * energia_obiettivo / fabbisogno * rng.normal(1, 0.12))
            pro_kg = max(0.8, rng.normal(1.6, 0.25))
            cho, pro = cho_kg * massa, pro_kg * massa
            grassi = max(25.0, (energia_obiettivo - 4 * cho - 4 * pro) / 9)
            energia = 4 * cho + 4 * pro + 9 * grassi
            fibra = max(5.0, 12 + 0.035 * cho + rng.normal(0, 4))

            cho_col = max(10.0, 0.25 * cho * rng.normal(1, 0.25))
            ig = float(np.clip(ig_preferito + rng.normal(0, 9), 30, 95))
            fibra_col = max(0.0, 0.2 * fibra * rng.normal(1, 0.35))
            sonno = float(np.clip(sonno_base + rng.normal(0, 0.7), 4.5, 10))
            picco = float(np.clip(
                88 + sensibilita * 0.42 * cho_col * ig / 65 - 1.1 * fibra_col
                + 2.5 * (7.5 - sonno) + rng.normal(0, 7), 70, 240))

            ea = (energia - dispendio) / magra
            # chi mangia sistematicamente poco perde peso: lo si vede sulla bilancia
            var_peso = -3.0 * restrizione + 0.15 + rng.normal(0, 0.25)
            idratazione = float(np.clip(1.8 + dispendio / 650 + rng.normal(0, 0.35), 0.8, 6))
            recupero = float(np.clip(
                62 + 5.0 * (sonno - 7.5) + 4.0 * (cho_kg - CHO_OBIETTIVO[tipo])
                + 6.0 * (min(pro_kg, 2.0) - 1.6) + 5.0 * (idratazione - 1.8 - dispendio / 650)
                + 0.25 * (ea - 40) - 0.004 * dispendio + rng.normal(0, 5), 0, 100))

            righe.append({
                "id_atleta": f"A{a + 1:02d}", "sport": sport, "sesso": sesso, "giorno": gno,
                "tipo_giornata": tipo, "eta": eta, "massa_kg": round(massa, 1),
                "massa_magra_kg": round(magra, 1), "energia_kcal": round(energia),
                "cho_g": round(cho), "pro_g": round(pro), "grassi_g": round(grassi),
                "fibra_g": round(fibra, 1), "cho_g_kg": round(cho_kg, 2),
                "pro_g_kg": round(pro_kg, 2), "cho_colazione_g": round(cho_col),
                "ig_colazione": round(ig), "fibra_colazione_g": round(fibra_col, 1),
                "dispendio_esercizio_kcal": round(dispendio),
                "variazione_peso_7g_kg": round(var_peso, 2),
                "disponibilita_energetica": round(ea, 1), "sonno_h": round(sonno, 1),
                "idratazione_l": round(idratazione, 2), "picco_glicemico": round(picco),
                "recupero": round(recupero, 1), "bassa_ea": int(ea < 30),
            })
    return pd.DataFrame(righe)


# =========================================================================
# ERRORI REALISTICI E PULIZIA
# =========================================================================

def colonne_misura(dominio: str, df: pd.DataFrame) -> List[str]:
    cat = CATALOGHI[dominio]
    return [c for c in df.columns if c in cat and cat[c].ruolo == "misura"]


def sporca(df: pd.DataFrame, dominio: str, quota_mancanti: float = 0.04,
           quota_errori: float = 0.02, quota_duplicati: float = 0.02,
           seed: int = 7) -> pd.DataFrame:
    """
    Aggiunge i tre difetti più comuni dei dati reali:
      · celle vuote (sensore scarico, questionario non compilato);
      · valori impossibili (GPS che "salta", virgola nel posto sbagliato);
      · righe duplicate (lo stesso file importato due volte).
    """
    rng = np.random.default_rng(seed + 1000)
    d = df.copy()
    misure = colonne_misura(dominio, d)
    cat = CATALOGHI[dominio]
    n = len(d)
    # celle vuote
    if quota_mancanti > 0:
        for c in misure:
            idx = rng.random(n) < quota_mancanti
            d.loc[idx, c] = np.nan
    # valori impossibili: ×10 oppure negativi
    if quota_errori > 0:
        quante = max(1, int(n * quota_errori))
        for _ in range(quante):
            r = int(rng.integers(0, n))
            c = misure[int(rng.integers(0, len(misure)))]
            v = d.at[d.index[r], c]
            if pd.isna(v):
                continue
            if rng.random() < 0.7:
                d.at[d.index[r], c] = float(v) * 10 if float(v) != 0 else (cat[c].massimo or 1) * 5
            else:
                d.at[d.index[r], c] = -abs(float(v)) - 1
    # duplicati
    if quota_duplicati > 0:
        k = int(n * quota_duplicati)
        if k:
            d = pd.concat([d, d.sample(k, random_state=seed)], ignore_index=True)
            d = d.sample(frac=1, random_state=seed).reset_index(drop=True)
    return d


def diagnosi(df: pd.DataFrame, dominio: str) -> pd.DataFrame:
    """Per ogni colonna: quante celle vuote e quanti valori fuori dai limiti plausibili."""
    cat = CATALOGHI[dominio]
    righe = []
    for c in df.columns:
        info = cat.get(c)
        vuote = int(df[c].isna().sum())
        fuori = 0
        if info is not None and info.minimo is not None and pd.api.types.is_numeric_dtype(df[c]):
            fuori = int(((df[c] < info.minimo) | (df[c] > info.massimo)).sum())
        righe.append({"colonna": c, "celle vuote": vuote, "valori impossibili": fuori})
    return pd.DataFrame(righe)


OPZIONI_MANCANTI = ["sostituisci con la mediana", "elimina le righe", "lascia così"]
OPZIONI_IMPOSSIBILI = ["trasforma in cella vuota (poi trattata come sopra)", "elimina le righe",
                       "lascia così"]


def pulisci(df: pd.DataFrame, dominio: str, togli_duplicati: bool = True,
            impossibili: str = OPZIONI_IMPOSSIBILI[0],
            mancanti: str = OPZIONI_MANCANTI[0]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce il dataset pulito e il registro di ciò che è stato fatto."""
    cat = CATALOGHI[dominio]
    d = df.copy()
    registro = []

    def nota(passo, dettaglio, righe_prima):
        registro.append({"passo": passo, "dettaglio": dettaglio,
                         "righe prima": righe_prima, "righe dopo": len(d)})

    if togli_duplicati:
        prima = len(d)
        d = d.drop_duplicates().reset_index(drop=True)
        nota("duplicati", f"{prima - len(d)} righe identiche eliminate", prima)

    if impossibili != "lascia così":
        prima = len(d)
        tot = 0
        maschera_righe = pd.Series(False, index=d.index)
        for c, info in cat.items():
            if c not in d.columns or info.minimo is None:
                continue
            fuori = (d[c] < info.minimo) | (d[c] > info.massimo)
            tot += int(fuori.sum())
            if impossibili.startswith("trasforma"):
                d.loc[fuori, c] = np.nan
            else:
                maschera_righe |= fuori
        if impossibili == "elimina le righe":
            d = d[~maschera_righe].reset_index(drop=True)
            nota("valori impossibili", f"{tot} valori fuori dai limiti: righe eliminate", prima)
        else:
            nota("valori impossibili", f"{tot} valori fuori dai limiti trasformati in celle vuote", prima)

    if mancanti != "lascia così":
        prima = len(d)
        vuote = int(d.isna().sum().sum())
        if mancanti == "elimina le righe":
            d = d.dropna().reset_index(drop=True)
            nota("celle vuote", f"{vuote} celle vuote: righe che le contenevano eliminate", prima)
        else:
            for c in d.columns:
                if d[c].isna().any() and pd.api.types.is_numeric_dtype(d[c]):
                    # mediana del singolo giocatore/atleta, se c'è; altrimenti di tutto il gruppo
                    gruppo = ID_GRUPPO[dominio]
                    d[c] = d[c].fillna(d.groupby(gruppo)[c].transform("median"))
                    d[c] = d[c].fillna(d[c].median())
            nota("celle vuote", f"{vuote} celle riempite con la mediana dello stesso atleta", prima)

    return d, pd.DataFrame(registro)


def genera(dominio: str, n: int, periodo: int, seed: int) -> pd.DataFrame:
    if dominio == "sport":
        return genera_sport(n, periodo, seed)
    return genera_nutrizione(n, periodo, seed)


# ###########################################################################
# 2 · DATI DELLO STUDENTE
# ###########################################################################
# dati_utente.py — Sport AI Lab Start · i dataset degli studenti
#
# Qui non si sa nulla in anticipo del dataset: nomi delle colonne, unità, tipi,
# separatori, decimali con la virgola, fogli Excel, celle vuote. Il modulo:
#
#     1. legge CSV (qualsiasi separatore, virgola o punto decimale), Excel e JSON;
#     2. converte in numeri le colonne che sono numeri scritti come testo
#        ("12,5", "1.200", "85%"), riconosce sì/no e date;
#     3. propone per ogni colonna un TIPO e un RUOLO (misura, categoria,
#        identificativo dell'atleta, ignora), che lo studente può correggere;
#     4. pulisce secondo le scelte dello studente, lasciando un registro.
#
# Il modulo non importa Streamlit.

RUOLI_COLONNA = ["misura", "categoria", "identificativo atleta", "ignora"]
TIPI = ["numero", "sì/no", "categoria", "testo", "data"]

_SI = {"si", "sì", "yes", "y", "true", "vero", "s", "1"}
_NO = {"no", "n", "false", "falso", "0"}
_PAROLE_ID = ("id", "codice", "cod", "atleta", "giocatore", "soggetto", "paziente", "player",
              "athlete", "subject", "nome", "name", "matricola")


# =========================================================================
# LETTURA
# =========================================================================

def fogli_excel(contenuto: bytes) -> List[str]:
    return pd.ExcelFile(io.BytesIO(contenuto)).sheet_names


def foglio_principale(contenuto: bytes) -> int:
    """L'indice del foglio con più celle piene: spesso il primo è una legenda."""
    fogli = pd.read_excel(io.BytesIO(contenuto), sheet_name=None, header=None, nrows=200)
    pieni = [int(f.notna().sum().sum()) for f in fogli.values()]
    return int(np.argmax(pieni)) if pieni else 0


def _testo(contenuto: bytes) -> str:
    for cod in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return contenuto.decode(cod)
        except UnicodeDecodeError:
            continue
    return contenuto.decode("utf-8", errors="replace")


def _indovina_separatore(testo: str) -> str:
    righe = [r for r in testo.splitlines()[:20] if r.strip()]
    migliore, punteggio = ",", -1
    for sep in (";", ",", "\t", "|"):
        conte = [r.count(sep) for r in righe]
        if not conte or min(conte) == 0:
            continue
        # il separatore giusto compare lo stesso numero di volte in ogni riga
        p = min(conte) * (2 if len(set(conte)) == 1 else 1)
        if p > punteggio:
            migliore, punteggio = sep, p
    return migliore


def leggi(contenuto: bytes, nome_file: str, foglio: str | None = None,
          separatore: str = "auto", riga_intestazione: int = 0) -> Tuple[pd.DataFrame, List[str]]:
    """Restituisce il DataFrame e le note su come è stato letto."""
    note = []
    est = nome_file.lower().rsplit(".", 1)[-1]
    if est in ("xlsx", "xlsm", "xls"):
        df = pd.read_excel(io.BytesIO(contenuto), sheet_name=foglio or 0, header=riga_intestazione)
        note.append(f"File Excel, foglio «{foglio or fogli_excel(contenuto)[0]}».")
    elif est == "json":
        dati = json.loads(_testo(contenuto))
        if isinstance(dati, dict):
            # {"colonna": [...]} oppure {"dati": [...]} oppure {"chiave": {...}}
            liste = [v for v in dati.values() if isinstance(v, list)]
            if len(liste) == 1 and liste[0] and isinstance(liste[0][0], dict):
                dati = liste[0]
        df = pd.json_normalize(dati) if isinstance(dati, list) else pd.DataFrame(dati)
        note.append("File JSON.")
    else:
        testo = _testo(contenuto)
        sep = _indovina_separatore(testo) if separatore == "auto" else separatore
        df = pd.read_csv(io.StringIO(testo), sep=sep, header=riga_intestazione,
                         dtype=str, keep_default_na=True, skipinitialspace=True)
        nomi = {";": "punto e virgola", ",": "virgola", "\t": "tabulazione", "|": "barra"}
        note.append(f"File di testo, separatore: {nomi.get(sep, repr(sep))}.")

    # righe e colonne completamente vuote, nomi ripuliti e unici
    prima = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = _nomi_unici([str(c).strip() or f"colonna_{i + 1}" for i, c in enumerate(df.columns)])
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")
                    or df[c].notna().any()]]
    if df.shape != prima:
        note.append(f"Tolte righe/colonne completamente vuote: da {prima[0]}×{prima[1]} "
                    f"a {df.shape[0]}×{df.shape[1]}.")
    df = df.reset_index(drop=True)
    df, conv = converti_tipi(df)
    note += conv
    return df, note


def _nomi_unici(nomi: List[str]) -> List[str]:
    visti: Dict[str, int] = {}
    out = []
    for n in nomi:
        if n in visti:
            visti[n] += 1
            out.append(f"{n}_{visti[n]}")
        else:
            visti[n] = 0
            out.append(n)
    return out


# =========================================================================
# CONVERSIONE DEI TIPI
# =========================================================================

def _come_numero(s: pd.Series) -> pd.Series:
    """'1.234,5' → 1234.5 · '12,5' → 12.5 · '85%' → 85 · ' 7 ' → 7."""
    t = s.astype(str).str.strip().str.replace(" ", "", regex=False)
    t = t.str.replace(r"[%€$\s]", "", regex=True)
    virgola = t.str.contains(",", regex=False)
    punto = t.str.contains(".", regex=False)
    # entrambi: il punto è il separatore delle migliaia (stile italiano)
    t = t.where(~(virgola & punto), t.str.replace(".", "", regex=False))
    t = t.str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def converti_tipi(df: pd.DataFrame, soglia: float = 0.9) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    note = []
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        piena = s.dropna()
        piena = piena[piena.astype(str).str.strip() != ""]
        if piena.empty:
            continue
        num = _come_numero(piena)
        if num.notna().mean() >= soglia:
            d[c] = _come_numero(s.where(s.astype(str).str.strip() != ""))
            persi = int(num.isna().sum())
            note.append(f"«{c}»: testo convertito in numero"
                        + (f" ({persi} valori non numerici diventano celle vuote)" if persi else "")
                        + ".")
            continue
        minus = piena.astype(str).str.strip().str.lower()
        if minus.isin(_SI | _NO).mean() >= 0.98 and minus.nunique() <= 2:
            d[c] = s.astype(str).str.strip().str.lower().map(
                lambda x: 1 if x in _SI else (0 if x in _NO else np.nan))
            note.append(f"«{c}»: sì/no convertito in 1/0.")
            continue
        if minus.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True).mean() >= soglia:
            date = pd.to_datetime(piena, errors="coerce", dayfirst=True, format="mixed")
            if date.notna().mean() >= soglia:
                d[c] = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
                note.append(f"«{c}»: riconosciuta come data.")
                continue
        d[c] = s.astype("string").str.strip().replace("", pd.NA).astype(object)
    return d, note


def tipo_di(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "data"
    if pd.api.types.is_bool_dtype(s):
        return "sì/no"
    if pd.api.types.is_numeric_dtype(s):
        valori = set(pd.unique(s.dropna()))
        return "sì/no" if valori and valori <= {0, 1} else "numero"
    n = s.nunique(dropna=True)
    lunghezza = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
    if n <= max(20, 0.05 * len(s)) and lunghezza < 40:
        return "categoria"
    return "testo"


# =========================================================================
# PROFILO E RUOLI
# =========================================================================

def profilo(df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per colonna: tipo rilevato, ruolo proposto, qualche numero utile."""
    righe = []
    id_trovato = False
    n = len(df)
    for c in df.columns:
        s = df[c]
        tipo = tipo_di(s)
        distinti = int(s.nunique(dropna=True))
        nome = c.lower()
        sembra_id = any(re.search(rf"(^|[_\s\-]){p}($|[_\s\-])", nome) or nome == p
                        for p in _PAROLE_ID)
        if tipo == "data":
            ruolo = "ignora"
        elif distinti <= 1:
            ruolo = "ignora"
        elif (not id_trovato and sembra_id and tipo in ("categoria", "testo", "numero")
              and 2 <= distinti < n):
            ruolo, id_trovato = "identificativo atleta", True
        elif tipo == "testo":
            ruolo = "ignora"
        elif tipo == "numero":
            ruolo = "misura"
        else:
            ruolo = "categoria"
        esempio = s.dropna().iloc[0] if s.notna().any() else ""
        riga = {"colonna": c, "tipo rilevato": tipo, "ruolo": ruolo, "significato": "",
                "unità": "", "min plausibile": np.nan, "max plausibile": np.nan,
                "valori distinti": distinti, "celle vuote %": round(100 * s.isna().mean(), 1),
                "esempio": str(esempio)[:30]}
        if tipo == "numero":
            riga["min"] = float(s.min()) if s.notna().any() else np.nan
            riga["max"] = float(s.max()) if s.notna().any() else np.nan
        righe.append(riga)
    out = pd.DataFrame(righe)
    for c in ("min plausibile", "max plausibile", "min", "max"):
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce").astype(float)
    return out


def colonna_id(schema: pd.DataFrame) -> str | None:
    ids = schema.loc[schema["ruolo"] == "identificativo atleta", "colonna"].tolist()
    return ids[0] if ids else None


def colonne_per_ruolo(schema: pd.DataFrame, *ruoli: str) -> List[str]:
    return schema.loc[schema["ruolo"].isin(ruoli), "colonna"].tolist()


def tabella_significato_utente(schema: pd.DataFrame) -> pd.DataFrame:
    return schema[["colonna", "tipo rilevato", "ruolo", "significato", "unità",
                   "min plausibile", "max plausibile"]].copy()


# =========================================================================
# DIAGNOSI E PULIZIA
# =========================================================================

def _confini_tukey(s: pd.Series, k: float = 3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr) if iqr > 0 else (None, None)


def diagnosi_utente(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for _, r in schema.iterrows():
        c = r["colonna"]
        if c not in df.columns:
            continue
        s = df[c]
        fuori = estremi = 0
        if pd.api.types.is_numeric_dtype(s) and r["ruolo"] == "misura":
            lo, hi = r.get("min plausibile"), r.get("max plausibile")
            if pd.notna(lo):
                fuori += int((s < float(lo)).sum())
            if pd.notna(hi):
                fuori += int((s > float(hi)).sum())
            a, b = _confini_tukey(s.dropna())
            if a is not None:
                estremi = int(((s < a) | (s > b)).sum())
        righe.append({"colonna": c, "ruolo": r["ruolo"], "celle vuote": int(s.isna().sum()),
                      "fuori dai limiti plausibili": fuori, "valori estremi (Tukey)": estremi})
    return pd.DataFrame(righe)


OPZ_FUORI = ["trasforma in cella vuota", "elimina le righe", "lascia così"]
OPZ_ESTREMI = ["trasforma in cella vuota", "elimina le righe", "lascia così"]
OPZ_VUOTE_NUM = ["sostituisci con la mediana", "elimina le righe", "lascia così"]
OPZ_VUOTE_CAT = ["sostituisci con la categoria più frequente", "crea la categoria «mancante»",
                 "elimina le righe", "lascia così"]


def pulisci_utente(df: pd.DataFrame, schema: pd.DataFrame, duplicati: bool = True,
            soglia_colonne_vuote: float = 60.0, fuori: str = OPZ_FUORI[0],
            estremi: str = "trasforma in cella vuota", vuote_num: str = OPZ_VUOTE_NUM[0],
            vuote_cat: str = OPZ_VUOTE_CAT[0]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restituisce dataset pulito, registro e schema aggiornato (colonne tolte)."""
    d = df.copy()
    sch = schema.copy()
    reg = []

    def nota(passo, dettaglio, prima):
        reg.append({"passo": passo, "dettaglio": dettaglio, "righe prima": prima,
                    "righe dopo": len(d)})

    if duplicati:
        prima = len(d)
        d = d.drop_duplicates().reset_index(drop=True)
        nota("righe duplicate", f"{prima - len(d)} eliminate", prima)

    usate = colonne_per_ruolo(sch, "misura", "categoria")
    troppo_vuote = [c for c in usate if c in d.columns and 100 * d[c].isna().mean() > soglia_colonne_vuote]
    if troppo_vuote:
        prima = len(d)
        d = d.drop(columns=troppo_vuote)
        sch.loc[sch["colonna"].isin(troppo_vuote), "ruolo"] = "ignora"
        nota("colonne quasi vuote", f"tolte (oltre {soglia_colonne_vuote:.0f}% vuote): "
             + ", ".join(troppo_vuote), prima)

    misure = [c for c in colonne_per_ruolo(sch, "misura")
              if c in d.columns and pd.api.types.is_numeric_dtype(d[c])]
    categorie = [c for c in colonne_per_ruolo(sch, "categoria") if c in d.columns]
    lim = sch.set_index("colonna")

    if fuori != "lascia così":
        prima = len(d)
        maschera = pd.Series(False, index=d.index)
        tot = 0
        for c in misure:
            lo, hi = lim.at[c, "min plausibile"], lim.at[c, "max plausibile"]
            m = pd.Series(False, index=d.index)
            if pd.notna(lo):
                m |= d[c] < float(lo)
            if pd.notna(hi):
                m |= d[c] > float(hi)
            tot += int(m.sum())
            if fuori.startswith("trasforma"):
                d.loc[m, c] = np.nan
            else:
                maschera |= m
        if fuori == "elimina le righe":
            d = d[~maschera].reset_index(drop=True)
        nota("fuori dai limiti plausibili", f"{tot} valori: {fuori}", prima)

    if estremi != "lascia così":
        prima = len(d)
        maschera = pd.Series(False, index=d.index)
        tot = 0
        for c in misure:
            a, b = _confini_tukey(d[c].dropna())
            if a is None:
                continue
            m = (d[c] < a) | (d[c] > b)
            tot += int(m.sum())
            if estremi.startswith("trasforma"):
                d.loc[m, c] = np.nan
            else:
                maschera |= m
        if estremi == "elimina le righe":
            d = d[~maschera].reset_index(drop=True)
        nota("valori estremi (Tukey, 3 IQR)", f"{tot} valori: {estremi}", prima)

    gruppo = colonna_id(sch)
    if vuote_num != "lascia così" and misure:
        prima = len(d)
        vuote = int(d[misure].isna().sum().sum())
        if vuote_num == "elimina le righe":
            d = d.dropna(subset=misure).reset_index(drop=True)
            nota("celle vuote (numeri)", f"{vuote}: righe eliminate", prima)
        else:
            for c in misure:
                if d[c].isna().any():
                    if gruppo and gruppo in d.columns:
                        d[c] = d[c].fillna(d.groupby(gruppo)[c].transform("median"))
                    d[c] = d[c].fillna(d[c].median())
            nota("celle vuote (numeri)", f"{vuote}: sostituite con la mediana"
                 + (" dello stesso atleta" if gruppo else ""), prima)

    if vuote_cat != "lascia così" and categorie:
        prima = len(d)
        vuote = int(d[categorie].isna().sum().sum())
        if vuote_cat == "elimina le righe":
            d = d.dropna(subset=categorie).reset_index(drop=True)
        else:
            for c in categorie:
                if d[c].isna().any():
                    if vuote_cat.startswith("crea"):
                        d[c] = d[c].astype(object).where(d[c].notna(), "mancante")
                    else:
                        moda = d[c].mode(dropna=True)
                        if len(moda):
                            d[c] = d[c].fillna(moda.iat[0])
        nota("celle vuote (categorie)", f"{vuote}: {vuote_cat}", prima)

    return d, pd.DataFrame(reg), sch


# ###########################################################################
# 3 · I CINQUE MODELLI
# ###########################################################################
# modelli.py — Sport AI Lab Start · i cinque modelli di machine learning
#
#     Regressione Lineare     bersaglio numerico
#     Regressione Logistica   bersaglio sì/no
#     Random Forest           bersaglio numerico o sì/no
#     XGBoost                 bersaglio numerico o sì/no
#     K-Means                 nessun bersaglio: cerca gruppi di atleti simili
#
# Ogni analisi restituisce sempre R² e MAE, anche quando il bersaglio è sì/no
# (calcolati sulla probabilità stimata) e anche per il K-Means (calcolati sulla
# distanza dai centri dei gruppi). Il significato esatto è spiegato in
# `didattica.py` e nella scheda Didattica dell'app.
#
# Il modulo non importa Streamlit.

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except Exception:                                   # xgboost non installato
    XGB_OK = False

COLORE = {"primario": "#1b5e5a", "secondario": "#c2603a", "neutro": "#8a9199",
          "chiaro": "#a8c8c4", "scuro": "#22303b"}
CICLO = ["#1b5e5a", "#c2603a", "#5b7fa6", "#b08c3f", "#7a5c8f", "#4f8f6b", "#a35a6d", "#3d4855"]

# compiti che ciascun modello sa svolgere
MODELLI: Dict[str, List[str]] = {
    "Regressione Lineare": ["regressione"],
    "Regressione Logistica": ["classificazione"],
    "Random Forest": ["regressione", "classificazione"],
    "XGBoost": ["regressione", "classificazione"],
    "K-Means": ["raggruppamento"],
}


def bersagli_per(modello: str, dominio: str) -> List[str]:
    """I bersagli che un modello può prevedere in questo dataset."""
    compiti = MODELLI[modello]
    out = []
    if "regressione" in compiti:
        out += bersagli(dominio, "numerico")
    if "classificazione" in compiti:
        out += bersagli(dominio, "binario")
    return out


def compito_di(dominio: str, bersaglio: str) -> str:
    return ("classificazione" if catalogo(dominio)[bersaglio].ruolo == "bersaglio sì/no"
            else "regressione")


# =========================================================================
# COSTRUZIONE
# =========================================================================

def _stimatore(modello: str, compito: str, seed: int, par: dict):
    alberi = int(par.get("alberi", 300))
    profondita = par.get("profondita")
    if modello == "Regressione Lineare":
        return LinearRegression()
    if modello == "Regressione Logistica":
        return LogisticRegression(max_iter=3000)
    if modello == "Random Forest":
        kw = dict(n_estimators=alberi, max_depth=profondita, min_samples_leaf=3,
                  random_state=seed, n_jobs=1)
        return RandomForestRegressor(**kw) if compito == "regressione" else RandomForestClassifier(**kw)
    if modello == "XGBoost":
        ritmo = float(par.get("ritmo", 0.05))
        prof = int(profondita or 3)
        if XGB_OK:
            kw = dict(n_estimators=alberi, max_depth=prof, learning_rate=ritmo, subsample=0.8,
                      colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0)
            return XGBRegressor(**kw) if compito == "regressione" else XGBClassifier(**kw)
        # ripiego: stesso principio (gradient boosting), libreria scikit-learn
        kw = dict(max_iter=alberi, max_depth=prof, learning_rate=ritmo, random_state=seed)
        return (HistGradientBoostingRegressor(**kw) if compito == "regressione"
                else HistGradientBoostingClassifier(**kw))
    raise ValueError(f"modello sconosciuto: {modello}")


def _preparazione(X: pd.DataFrame) -> ColumnTransformer:
    """Numeriche: mediana per le celle vuote residue + standardizzazione.
    Categoriche: una colonna 0/1 per ogni categoria (one-hot)."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parti = []
    if num:
        parti.append(("num", Pipeline([("vuote", SimpleImputer(strategy="median")),
                                       ("scala", StandardScaler())]), num))
    if cat:
        parti.append(("cat", Pipeline([("vuote", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parti, verbose_feature_names_out=False)


def _dividi(df: pd.DataFrame, gruppo: str, quota: float, seed: int):
    """Divisione PER ATLETA: chi sta nella verifica non è mai stato visto in
    addestramento. È la domanda onesta: il modello funziona su una persona nuova?"""
    gss = GroupShuffleSplit(n_splits=1, test_size=quota, random_state=seed)
    tr, te = next(gss.split(df, groups=df[gruppo]))
    return df.iloc[tr], df.iloc[te]


def _importanza(pipe: Pipeline, colonne: List[str]) -> pd.DataFrame | None:
    """Importanza riportata alle colonne originali (le categorie one-hot sono sommate)."""
    prep, mod = pipe.named_steps["prep"], pipe.named_steps["mod"]
    nomi = list(prep.get_feature_names_out())
    if hasattr(mod, "feature_importances_"):
        valori = np.asarray(mod.feature_importances_, dtype=float)
    elif hasattr(mod, "coef_"):
        valori = np.abs(np.ravel(mod.coef_))
    else:
        return None
    serie = pd.Series(0.0, index=colonne)
    for n, v in zip(nomi, valori):
        origine = n if n in colonne else next(
            (c for c in colonne if n.startswith(c + "_")), n)
        serie[origine] = serie.get(origine, 0.0) + v
    if serie.sum() > 0:
        serie = serie / serie.sum() * 100
    return (serie.sort_values(ascending=False).rename("importanza %").round(1)
            .reset_index().rename(columns={"index": "variabile"}))


def _coefficienti(pipe: Pipeline) -> pd.DataFrame | None:
    """Per i modelli lineari: segno e peso di ogni variabile (dati standardizzati)."""
    mod = pipe.named_steps["mod"]
    if not hasattr(mod, "coef_"):
        return None
    nomi = list(pipe.named_steps["prep"].get_feature_names_out())
    c = np.ravel(mod.coef_)
    t = pd.DataFrame({"variabile": nomi, "coefficiente": np.round(c, 3)})
    t["effetto"] = np.where(c > 0, "↑ fa salire", "↓ fa scendere")
    return t.reindex(t["coefficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# =========================================================================
# ADDESTRAMENTO E MISURE
# =========================================================================

def addestra(df: pd.DataFrame, dominio: str, modello: str, bersaglio: str,
             predittori: List[str], seed: int = 7, quota: float = 0.25,
             soglia: float = 0.5, par: dict | None = None) -> dict:
    par = par or {}
    if not predittori:
        raise ValueError("Scegli almeno una variabile in ingresso.")
    compito = compito_di(dominio, bersaglio)
    if compito not in MODELLI[modello]:
        raise ValueError(f"{modello} non può prevedere '{bersaglio}'.")
    gruppo = ID_GRUPPO[dominio]
    d = df.dropna(subset=[bersaglio]).reset_index(drop=True)
    tr, te = _dividi(d, gruppo, quota, seed)
    if compito == "classificazione" and tr[bersaglio].nunique() < 2:
        raise ValueError("Nell'insieme di addestramento c'è una sola classe: genera più dati.")

    pipe = Pipeline([("prep", _preparazione(tr[predittori])),
                     ("mod", _stimatore(modello, compito, seed, par))])
    y_tr = tr[bersaglio].astype(float).values
    y_te = te[bersaglio].astype(float).values
    pipe.fit(tr[predittori], y_tr.astype(int) if compito == "classificazione" else y_tr)

    ris = {"compito": compito, "modello": modello, "bersaglio": bersaglio,
           "predittori": list(predittori), "n_addestramento": len(tr), "n_verifica": len(te),
           "atleti_verifica": sorted(te[gruppo].unique().tolist()),
           "celle_vuote_residue": int(d[predittori].isna().sum().sum()),
           "xgb_sostituito": modello == "XGBoost" and not XGB_OK}

    if compito == "regressione":
        p_tr, p_te = pipe.predict(tr[predittori]), pipe.predict(te[predittori])
        media = float(np.mean(y_tr))
        ris["misure"] = {
            "R² (verifica)": r2_score(y_te, p_te),
            "MAE (verifica)": mean_absolute_error(y_te, p_te),
            "R² (addestramento)": r2_score(y_tr, p_tr),
            "MAE (addestramento)": mean_absolute_error(y_tr, p_tr),
        }
        ris["riferimento"] = {"R² (verifica)": r2_score(y_te, np.full_like(y_te, media)),
                              "MAE (verifica)": mean_absolute_error(y_te, np.full_like(y_te, media))}
        stimato = p_te
    else:
        pr_tr = pipe.predict_proba(tr[predittori])[:, 1]
        pr_te = pipe.predict_proba(te[predittori])[:, 1]
        classe = (pr_te >= soglia).astype(int)
        frequenza = float(np.mean(y_tr))
        positivi = y_te == 1
        ris["misure"] = {
            "R² (verifica)": r2_score(y_te, pr_te),
            "MAE (verifica)": mean_absolute_error(y_te, pr_te),
            "R² (addestramento)": r2_score(y_tr, pr_tr),
            "MAE (addestramento)": mean_absolute_error(y_tr, pr_tr),
            "Accuratezza": accuracy_score(y_te, classe),
            "Sensibilità (eventi riconosciuti)": float(classe[positivi].mean()) if positivi.any() else np.nan,
            "Specificità (non-eventi riconosciuti)": float((1 - classe[~positivi]).mean()) if (~positivi).any() else np.nan,
            "AUC": roc_auc_score(y_te, pr_te) if len(np.unique(y_te)) == 2 else np.nan,
        }
        base = np.full_like(y_te, frequenza)
        ris["riferimento"] = {"R² (verifica)": r2_score(y_te, base),
                              "MAE (verifica)": mean_absolute_error(y_te, base),
                              "Accuratezza": accuracy_score(y_te, np.zeros_like(y_te)),
                              "AUC": 0.5}
        ris["soglia"] = soglia
        ris["frequenza_evento"] = frequenza
        ris["confusione"] = confusion_matrix(y_te.astype(int), classe, labels=[0, 1])
        stimato = pr_te

    tabella = te[[gruppo] + [c for c in ("settimana", "giorno") if c in te.columns]].copy()
    tabella["osservato"] = y_te
    tabella["stimato" if compito == "regressione" else "probabilità stimata"] = np.round(stimato, 3)
    if compito == "regressione":
        tabella["errore"] = np.round(stimato - y_te, 2)
    else:
        tabella["allarme"] = (stimato >= soglia).astype(int)
    ris["tabella"] = tabella.reset_index(drop=True)
    ris["importanza"] = _importanza(pipe, list(predittori))
    ris["coefficienti"] = _coefficienti(pipe) if modello.startswith("Regressione") else None
    return ris


def confronta_tutti(df: pd.DataFrame, dominio: str, bersaglio: str, predittori: List[str],
                    seed: int = 7, quota: float = 0.25) -> pd.DataFrame:
    """Tutti i modelli compatibili, stesso bersaglio, stessa divisione."""
    compito = compito_di(dominio, bersaglio)
    righe = []
    for nome, compiti in MODELLI.items():
        if compito not in compiti:
            continue
        r = addestra(df, dominio, nome, bersaglio, predittori, seed=seed, quota=quota)
        m = r["misure"]
        riga = {"modello": nome,
                "R² verifica": round(m["R² (verifica)"], 3),
                "MAE verifica": round(m["MAE (verifica)"], 3),
                "R² addestramento": round(m["R² (addestramento)"], 3),
                "MAE addestramento": round(m["MAE (addestramento)"], 3)}
        if compito == "classificazione":
            riga["AUC"] = round(m["AUC"], 3)
        righe.append(riga)
    rif = r["riferimento"]
    righe.append({"modello": "riferimento banale (media)",
                  "R² verifica": round(rif["R² (verifica)"], 3),
                  "MAE verifica": round(rif["MAE (verifica)"], 3),
                  "R² addestramento": np.nan, "MAE addestramento": np.nan,
                  **({"AUC": 0.5} if compito == "classificazione" else {})})
    return pd.DataFrame(righe)


# =========================================================================
# K-MEANS
# =========================================================================

def profili_medi(df: pd.DataFrame, dominio: str) -> pd.DataFrame:
    """Una riga per atleta: media delle misure, categoria più frequente."""
    gruppo = ID_GRUPPO[dominio]
    num = [c for c in predittori(dominio, df, None) if pd.api.types.is_numeric_dtype(df[c])]
    num += [b for b in bersagli(dominio, "numerico") + bersagli(dominio, "binario")
            if b in df.columns]
    medie = df.groupby(gruppo)[num].mean().round(2)
    cat = [c for c in df.columns if catalogo(dominio).get(c) is not None
           and catalogo(dominio)[c].ruolo == "categoria" and c != "tipo_giornata"]
    if cat:
        medie = df.groupby(gruppo)[cat].agg(lambda s: s.mode().iat[0]).join(medie)
    return medie.reset_index()


def raggruppa(df: pd.DataFrame, dominio: str, variabili: List[str], k: int = 3,
              seed: int = 7) -> dict:
    if len(variabili) < 2:
        raise ValueError("Scegli almeno due variabili.")
    prof = profili_medi(df, dominio)
    X = prof[variabili].astype(float)
    X = X.fillna(X.median())
    Z = StandardScaler().fit_transform(X)
    if k >= len(prof):
        raise ValueError("Servono più atleti che gruppi.")
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    totale = float(((Z - Z.mean(axis=0)) ** 2).sum())
    r2 = 1 - km.inertia_ / totale
    mae = float(np.mean(np.abs(Z - km.cluster_centers_[km.labels_])))

    gomito = []
    for kk in range(2, min(9, len(prof))):
        m = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        gomito.append({"k": kk, "R²": 1 - m.inertia_ / totale,
                       "MAE": float(np.mean(np.abs(Z - m.cluster_centers_[m.labels_])))})

    prof = prof.copy()
    prof["gruppo"] = [f"Gruppo {l + 1}" for l in km.labels_]
    xy = PCA(n_components=2, random_state=seed).fit_transform(Z)
    prof["pca_1"], prof["pca_2"] = xy[:, 0], xy[:, 1]
    identikit = prof.groupby("gruppo")[variabili].mean().round(2)
    identikit.insert(0, "atleti", prof.groupby("gruppo").size())
    composizione = None
    cat = "ruolo" if dominio == "sport" else "sport"
    if cat in prof.columns:
        composizione = pd.crosstab(prof["gruppo"], prof[cat])
    return {"compito": "raggruppamento", "modello": "K-Means", "k": k, "variabili": variabili,
            "misure": {"R² (varianza spiegata dai gruppi)": r2,
                       "MAE (distanza media dal centro, in dev. standard)": mae},
            "gomito": pd.DataFrame(gomito), "tabella": prof, "identikit": identikit.reset_index(),
            "composizione": composizione, "categoria": cat}


# =========================================================================
# GRAFICI
# =========================================================================

def _asse(ax, titolo, x="", y=""):
    ax.set_title(titolo, fontsize=10.5, color=COLORE["scuro"], pad=9)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(y, fontsize=9)
    ax.tick_params(labelsize=8.5)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.7)


def g_osservato_stimato(ris: dict) -> plt.Figure:
    t = ris["tabella"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(t["osservato"], t["stimato"], s=18, alpha=0.6, color=COLORE["primario"])
    lo = float(min(t["osservato"].min(), t["stimato"].min()))
    hi = float(max(t["osservato"].max(), t["stimato"].max()))
    ax.plot([lo, hi], [lo, hi], color=COLORE["secondario"], linewidth=1.5, linestyle="--",
            label="previsione perfetta")
    ax.legend(fontsize=8, frameon=False)
    _asse(ax, "Osservato contro stimato (dati di verifica)", "valore osservato", "valore stimato")
    fig.tight_layout()
    return fig


def g_importanza(ris: dict, quante: int = 12) -> plt.Figure | None:
    imp = ris.get("importanza")
    if imp is None or not len(imp):
        return None
    d = imp.head(quante).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(d) + 1.4))
    ax.barh(d["variabile"], d["importanza %"], color=COLORE["primario"])
    _asse(ax, "Quanto pesa ogni variabile", "importanza relativa (%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def g_confusione(ris: dict) -> plt.Figure:
    m = ris["confusione"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(m, cmap="BuGn")
    nomi = [["veri negativi", "falsi allarmi"], ["eventi mancati", "eventi presi"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n{nomi[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > m.max() * 0.55 else COLORE["scuro"])
    ax.set_xticks([0, 1], ["previsto 0", "previsto 1"], fontsize=8.5)
    ax.set_yticks([0, 1], ["reale 0", "reale 1"], fontsize=8.5)
    ax.set_title(f"Matrice di confusione (soglia {ris['soglia']:.2f})", fontsize=10.5,
                 color=COLORE["scuro"])
    fig.tight_layout()
    return fig


def g_probabilita(ris: dict) -> plt.Figure:
    t = ris["tabella"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bins = np.linspace(0, 1, 21)
    ax.hist(t.loc[t["osservato"] == 0, "probabilità stimata"], bins=bins, alpha=0.7,
            color=COLORE["neutro"], label="nessun evento")
    ax.hist(t.loc[t["osservato"] == 1, "probabilità stimata"], bins=bins, alpha=0.8,
            color=COLORE["secondario"], label="evento avvenuto")
    ax.axvline(ris["soglia"], color=COLORE["scuro"], linestyle="--", linewidth=1.2,
               label="soglia di allarme")
    ax.legend(fontsize=8, frameon=False)
    _asse(ax, "Probabilità assegnate dal modello", "probabilità stimata", "righe")
    fig.tight_layout()
    return fig


def g_mappa_gruppi(ris: dict) -> plt.Figure:
    t = ris["tabella"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, (g, blocco) in enumerate(t.groupby("gruppo")):
        ax.scatter(blocco["pca_1"], blocco["pca_2"], s=60, color=CICLO[i % len(CICLO)],
                   label=g, alpha=0.85, edgecolor="white")
    _asse(ax, "Mappa degli atleti (2 dimensioni riassuntive)", "componente 1", "componente 2")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def g_gomito(ris: dict) -> plt.Figure:
    g = ris["gomito"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(g["k"], g["R²"], marker="o", color=COLORE["primario"], linewidth=2)
    ax.axvline(ris["k"], color=COLORE["secondario"], linestyle="--", linewidth=1.2,
               label=f"k scelto = {ris['k']}")
    ax.legend(fontsize=8, frameon=False)
    _asse(ax, "Metodo del gomito: quanti gruppi?", "numero di gruppi (k)", "R² (varianza spiegata)")
    fig.tight_layout()
    return fig


def g_confronto(tab: pd.DataFrame) -> plt.Figure:
    d = tab.iloc[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 0.45 * len(d) + 1.6))
    colori = [COLORE["neutro"] if "riferimento" in m else COLORE["primario"] for m in d["modello"]]
    axs[0].barh(d["modello"], d["R² verifica"], color=colori)
    _asse(axs[0], "R² in verifica (più alto è meglio)")
    axs[0].axvline(0, color=COLORE["scuro"], linewidth=0.8)
    axs[1].barh(d["modello"], d["MAE verifica"], color=colori)
    axs[1].set_yticklabels([])
    _asse(axs[1], "MAE in verifica (più basso è meglio)")
    for a in axs:
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# =========================================================================
# LETTURA AUTOMATICA DEL RISULTATO
# =========================================================================

def commenta(ris: dict, unita: str = "") -> List[str]:
    """Due o tre frasi che traducono i numeri in parole."""
    m = ris["misure"]
    out = []
    if ris["compito"] == "raggruppamento":
        r2 = list(m.values())[0]
        out.append(f"I {ris['k']} gruppi spiegano il <b>{r2 * 100:.0f}%</b> delle differenze fra "
                   f"atleti sulle variabili scelte.")
        out.append("Guarda il grafico del gomito: se aggiungendo un gruppo l'R² sale di poco, "
                   "il gruppo in più non serve.")
        out.append("I gruppi non hanno un nome: lo dai tu, leggendo l'identikit.")
        return out

    r2, mae = m["R² (verifica)"], m["MAE (verifica)"]
    r2_tr = m["R² (addestramento)"]
    rif_mae = ris["riferimento"]["MAE (verifica)"]
    if ris["compito"] == "regressione":
        out.append(f"Sugli atleti mai visti il modello sbaglia in media di <b>{mae:.2f} {unita}</b> "
                   f"(usando sempre la media si sbaglierebbe di {rif_mae:.2f}).")
        if r2 >= 0.7:
            giud = "coglie bene il fenomeno"
        elif r2 >= 0.3:
            giud = "è utile, ma molta variabilità resta inspiegata"
        elif r2 >= 0:
            giud = "è debole: il bersaglio dipende da cose che non misuriamo"
        else:
            giud = "fa peggio della media: non va usato"
        out.append(f"R² in verifica = <b>{r2:.2f}</b>: il modello {giud}.")
    else:
        auc = m.get("AUC", np.nan)
        out.append(f"AUC = <b>{auc:.2f}</b> (0,5 = tirare a moneta, 1 = perfetto). "
                   f"Con la soglia {ris['soglia']:.2f} l'allarme prende il "
                   f"<b>{m['Sensibilità (eventi riconosciuti)'] * 100:.0f}%</b> degli eventi veri.")
        rif_r2 = ris["riferimento"]["R² (verifica)"]
        out.append(f"R² sulle probabilità = <b>{r2:.2f}</b>, MAE = <b>{mae:.2f}</b> (dare a tutti la "
                   f"stessa probabilità, la frequenza dell'evento, darebbe R² = {rif_r2:.2f} e "
                   f"MAE = {rif_mae:.2f}). Con eventi rari e poche persone in verifica l'R² può "
                   f"essere vicino a zero o negativo: conta soprattutto il confronto con il "
                   f"riferimento e l'AUC.")
    if r2_tr - r2 > 0.3:
        out.append(f"⚠️ R² in addestramento = {r2_tr:.2f} contro {r2:.2f} in verifica: il modello "
                   f"ha in parte <b>imparato a memoria</b> (sovradattamento).")
    if ris.get("celle_vuote_residue"):
        out.append(f"Nota: c'erano ancora {ris['celle_vuote_residue']} celle vuote, riempite con la "
                   f"mediana. Meglio pulire il dataset al passo 2.")
    return out


# ###########################################################################
# 4 · MODELLI SU UN DATASET QUALSIASI
# ###########################################################################
# ml_generico.py — Sport AI Lab Start · machine learning su un dataset qualsiasi
#
# Gli stessi cinque modelli delle sezioni KPI, ma senza sapere in anticipo quali
# colonne ci sono. Il modello si adatta al dataset:
#
#     · il COMPITO si decide dal bersaglio: numero con molti valori → regressione;
#       sì/no o categoria → classificazione, anche con più di due classi;
#     · le colonne categoriali diventano colonne 0/1, le numeriche si standardizzano,
#       le celle vuote residue si riempiono (mediana o categoria più frequente);
#     · la divisione è PER ATLETA se c'è una colonna identificativa con abbastanza
#       persone, altrimenti casuale (stratificata per le classi);
#     · le classi troppo rare vengono tolte, i dataset troppo piccoli segnalati;
#     · le colonne quasi identiche al bersaglio vengono segnalate come possibile
#       fuga di informazione.
#
# R² e MAE ci sono sempre:
#     regressione       sui valori
#     classificazione   sulle probabilità (MAE = 1 − probabilità data alla classe vera)
#     K-Means           sulla distanza dai centri dei gruppi

MAX_CLASSI = 15


# =========================================================================
# BERSAGLI E COMPITI
# =========================================================================

def compito_suggerito(s: pd.Series) -> str:
    """Numero con più di 10 valori diversi → regressione; altrimenti classificazione."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
        return "regressione"
    return "classificazione"


def bersagli_possibili(df: pd.DataFrame, candidate: List[str], modello: str) -> List[str]:
    compiti = MODELLI[modello]
    out = []
    for c in candidate:
        s = df[c].dropna()
        n = s.nunique()
        if n < 2:
            continue
        num = pd.api.types.is_numeric_dtype(s)
        if "regressione" in compiti and num and n > 2:
            out.append(c)
        elif "classificazione" in compiti and 2 <= n <= MAX_CLASSI:
            out.append(c)
    return out


def compiti_ammessi(df: pd.DataFrame, bersaglio: str, modello: str) -> List[str]:
    s = df[bersaglio].dropna()
    n = s.nunique()
    out = []
    if "regressione" in MODELLI[modello] and pd.api.types.is_numeric_dtype(s) and n > 2:
        out.append("regressione")
    if "classificazione" in MODELLI[modello] and 2 <= n <= MAX_CLASSI:
        out.append("classificazione")
    return out


def avvisi_fuga(df: pd.DataFrame, bersaglio: str, predittori: List[str]) -> List[str]:
    """Colonne quasi identiche al bersaglio: probabilmente contengono la risposta."""
    avvisi = []
    y = df[bersaglio]
    y_num = pd.api.types.is_numeric_dtype(y)
    for c in predittori:
        x = df[c]
        if y_num and pd.api.types.is_numeric_dtype(x):
            coppia = pd.concat([x, y], axis=1).dropna()
            if len(coppia) > 5 and coppia.iloc[:, 0].std() > 0 and coppia.iloc[:, 1].std() > 0:
                r = coppia.iloc[:, 0].corr(coppia.iloc[:, 1])
                if abs(r) >= 0.95:
                    avvisi.append(f"**{c}** ha correlazione {r:+.2f} con il bersaglio: contiene "
                                  "la risposta, o è calcolata a partire da essa?")
        else:
            # una categoria che determina il bersaglio quasi senza eccezioni
            coppia = pd.concat([x.astype(str), y.astype(str)], axis=1).dropna()
            if x.nunique() > 1 and len(coppia) > 10:
                purezza = coppia.groupby(coppia.columns[0])[coppia.columns[1]].agg(
                    lambda s: s.value_counts(normalize=True).iat[0]).mean()
                if purezza >= 0.99 and x.nunique() < 0.5 * len(coppia):
                    avvisi.append(f"**{c}** determina il bersaglio quasi senza eccezioni: "
                                  "verifica che non sia nota solo dopo l'evento.")
    return avvisi


# =========================================================================
# DIVISIONE
# =========================================================================

def _dividi_generico(d: pd.DataFrame, y: np.ndarray, gruppo: str | None, divisione: str, quota: float,
            seed: int, classificazione: bool):
    if divisione == "per atleta" and gruppo:
        gss = GroupShuffleSplit(n_splits=1, test_size=quota, random_state=seed)
        tr, te = next(gss.split(d, groups=d[gruppo].astype(str)))
        return tr, te
    idx = np.arange(len(d))
    strat = None
    if classificazione:
        _, conte = np.unique(y, return_counts=True)
        if conte.min() >= 2:
            strat = y
    tr, te = train_test_split(idx, test_size=quota, random_state=seed, stratify=strat)
    return tr, te


# =========================================================================
# ADDESTRAMENTO
# =========================================================================

def addestra_generico(df: pd.DataFrame, modello: str, bersaglio: str, compito: str,
             predittori: List[str], categoriali: List[str], gruppo: str | None = None,
             divisione: str = "casuale", quota: float = 0.25, seed: int = 7,
             soglia: float = 0.5, par: dict | None = None, classe_positiva=None) -> dict:
    par = par or {}
    predittori = [p for p in predittori if p != bersaglio]
    if not predittori:
        raise ValueError("Scegli almeno una variabile in ingresso.")
    if compito not in MODELLI[modello]:
        raise ValueError(f"{modello} non svolge il compito «{compito}».")
    avvisi = []
    d = df.dropna(subset=[bersaglio]).reset_index(drop=True)
    tolte = len(df) - len(d)
    if tolte:
        avvisi.append(f"{tolte} righe senza bersaglio sono state escluse.")

    X = d[predittori].copy()
    for c in predittori:
        if c in categoriali or not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    classi = None
    if compito == "classificazione":
        etichette = d[bersaglio].astype(str)
        conte = etichette.value_counts()
        rare = conte[conte < 5].index.tolist()
        if rare and len(conte) - len(rare) >= 2:
            tieni = ~etichette.isin(rare)
            avvisi.append(f"Classi con meno di 5 casi escluse: {', '.join(map(str, rare))}.")
            d, X, etichette = d[tieni].reset_index(drop=True), X[tieni].reset_index(drop=True), \
                etichette[tieni].reset_index(drop=True)
        classi = sorted(etichette.unique(), key=lambda v: (len(v), v))
        if len(classi) < 2:
            raise ValueError("Il bersaglio ha una sola classe: non c'è nulla da prevedere.")
        if len(classi) == 2:
            if classe_positiva is None or str(classe_positiva) not in classi:
                # di default: «1», «sì», «true», altrimenti la classe più rara
                noti = [c for c in classi if c.lower() in ("1", "1.0", "sì", "si", "yes", "true")]
                classe_positiva = noti[0] if noti else etichette.value_counts().index[-1]
            classi = [c for c in classi if c != str(classe_positiva)] + [str(classe_positiva)]
        mappa = {c: i for i, c in enumerate(classi)}
        y = etichette.map(mappa).to_numpy()
    else:
        y = pd.to_numeric(d[bersaglio], errors="coerce").to_numpy(dtype=float)

    if len(d) < 20:
        raise ValueError(f"Solo {len(d)} righe utilizzabili: troppo poche per addestrare e "
                         "verificare un modello (ne servono almeno 20, meglio 100).")

    n_gruppi = d[gruppo].nunique() if gruppo else 0
    if divisione == "per atleta" and (not gruppo or n_gruppi < 4):
        avvisi.append("Servono almeno 4 atleti diversi per la divisione per atleta: uso quella "
                      "casuale.")
        divisione = "casuale"
    tr, te = _dividi_generico(d, y, gruppo, divisione, quota, seed, compito == "classificazione")
    if (compito == "classificazione" and divisione == "per atleta"
            and len(np.unique(y[tr])) < len(classi)):
        # succede quando la classe è una caratteristica fissa della persona (es. il ruolo)
        avvisi.append("Con la divisione per atleta alcune classi finivano tutte nella verifica "
                      "(il bersaglio è probabilmente una caratteristica fissa della persona): "
                      "uso la divisione casuale.")
        divisione = "casuale"
        tr, te = _dividi_generico(d, y, gruppo, divisione, quota, seed, True)
    if compito == "classificazione" and len(np.unique(y[tr])) < len(classi):
        raise ValueError("Nell'insieme di addestramento manca almeno una classe: servono più "
                         "righe per le classi rare.")

    pipe = Pipeline([("prep", _preparazione(X.iloc[tr])),
                     ("mod", _stimatore(modello, compito, seed, par))])
    pipe.fit(X.iloc[tr], y[tr])

    ris = {"compito": compito, "modello": modello, "bersaglio": bersaglio,
           "predittori": predittori, "n_addestramento": len(tr), "n_verifica": len(te),
           "divisione": divisione, "avvisi": avvisi,
           "celle_vuote_residue": int(X.isna().sum().sum()),
           "xgb_sostituito": modello == "XGBoost" and not XGB_OK}
    if gruppo and divisione == "per atleta":
        ris["atleti_verifica"] = sorted(map(str, d[gruppo].iloc[te].unique()))

    if compito == "regressione":
        p_tr, p_te = pipe.predict(X.iloc[tr]), pipe.predict(X.iloc[te])
        y_tr, y_te = y[tr], y[te]
        media = float(np.mean(y_tr))
        ris["misure"] = {"R² (verifica)": r2_score(y_te, p_te),
                         "MAE (verifica)": mean_absolute_error(y_te, p_te),
                         "R² (addestramento)": r2_score(y_tr, p_tr),
                         "MAE (addestramento)": mean_absolute_error(y_tr, p_tr)}
        base = np.full_like(y_te, media)
        ris["riferimento"] = {"R² (verifica)": r2_score(y_te, base),
                              "MAE (verifica)": mean_absolute_error(y_te, base)}
        tab = pd.DataFrame({"osservato": y_te, "stimato": np.round(p_te, 3),
                            "errore": np.round(p_te - y_te, 3)})
    else:
        P_tr, P_te = pipe.predict_proba(X.iloc[tr]), pipe.predict_proba(X.iloc[te])
        k = len(classi)
        r2_tr, mae_tr = _r2_mae_prob(y[tr], P_tr, k)
        r2_te, mae_te = _r2_mae_prob(y[te], P_te, k)
        freq = np.bincount(y[tr], minlength=k) / len(tr)
        base = np.tile(freq, (len(te), 1))
        r2_b, mae_b = _r2_mae_prob(y[te], base, k)
        if k == 2:
            pred = (P_te[:, 1] >= soglia).astype(int)
        else:
            pred = P_te.argmax(axis=1)
        y_te = y[te]
        misure = {"R² (verifica)": r2_te, "MAE (verifica)": mae_te,
                  "R² (addestramento)": r2_tr, "MAE (addestramento)": mae_tr,
                  "Accuratezza": accuracy_score(y_te, pred),
                  "Accuratezza bilanciata": balanced_accuracy_score(y_te, pred)}
        # AUC «una classe contro le altre», media sulle classi presenti in verifica
        auc = [roc_auc_score(y_te == j, P_te[:, j]) for j in (range(1, 2) if k == 2 else range(k))
               if 0 < np.sum(y_te == j) < len(y_te)]
        misure["AUC"] = float(np.mean(auc)) if auc else np.nan
        if k == 2:
            pos = y_te == 1
            misure["Sensibilità (eventi riconosciuti)"] = float(pred[pos].mean()) if pos.any() else np.nan
            misure["Specificità (non-eventi riconosciuti)"] = float((1 - pred[~pos]).mean()) if (~pos).any() else np.nan
        ris["misure"] = misure
        ris["riferimento"] = {"R² (verifica)": r2_b, "MAE (verifica)": mae_b,
                              "Accuratezza": float(np.mean(y_te == np.argmax(freq))), "AUC": 0.5}
        ris["classi"] = classi
        ris["soglia"] = soglia
        ris["confusione"] = confusione = confusion_matrix(y_te, pred, labels=list(range(k)))
        tab = pd.DataFrame({"osservato": [classi[i] for i in y_te],
                            "previsto": [classi[i] for i in pred]})
        if k == 2:
            tab["probabilità stimata"] = np.round(P_te[:, 1], 3)
            ris["classe_positiva"] = classi[1]
        else:
            tab["probabilità della classe vera"] = np.round(P_te[np.arange(len(te)), y_te], 3)
    if gruppo:
        tab.insert(0, gruppo, d[gruppo].iloc[te].astype(str).to_numpy())
    ris["tabella"] = tab
    ris["importanza"] = _importanza(pipe, predittori)
    ris["coefficienti"] = (_coefficienti(pipe) if modello.startswith("Regressione")
                           and not (compito == "classificazione" and len(classi) > 2) else None)
    return ris


def _r2_mae_prob(y: np.ndarray, P: np.ndarray, k: int):
    """R² fra classi vere (0/1 per colonna) e probabilità; MAE = 1 − p(classe vera)."""
    Y = np.eye(k)[y]
    if k == 2:                                   # come nelle sezioni KPI: sulla classe positiva
        Y, P = Y[:, 1:], P[:, 1:]
    ss_res = np.sum((Y - P) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if k == 2:
        mae = float(np.mean(np.abs(Y - P)))
    else:
        mae = float(np.mean(1 - np.sum(Y * P, axis=1)))
    return float(r2), mae


def confronta_tutti_generico(df, bersaglio, compito, predittori, categoriali, gruppo, divisione,
                    quota=0.25, seed=7) -> pd.DataFrame:
    righe = []
    ultimo = None
    for nome, compiti in MODELLI.items():
        if compito not in compiti:
            continue
        r = addestra_generico(df, nome, bersaglio, compito, predittori, categoriali, gruppo, divisione,
                     quota, seed)
        ultimo = r
        m = r["misure"]
        riga = {"modello": nome, "R² verifica": round(m["R² (verifica)"], 3),
                "MAE verifica": round(m["MAE (verifica)"], 3),
                "R² addestramento": round(m["R² (addestramento)"], 3),
                "MAE addestramento": round(m["MAE (addestramento)"], 3)}
        if compito == "classificazione":
            riga["AUC"] = round(m["AUC"], 3) if pd.notna(m["AUC"]) else np.nan
            riga["accuratezza bilanciata"] = round(m["Accuratezza bilanciata"], 3)
        righe.append(riga)
    rif = ultimo["riferimento"]
    righe.append({"modello": "riferimento banale (media / frequenze)",
                  "R² verifica": round(rif["R² (verifica)"], 3),
                  "MAE verifica": round(rif["MAE (verifica)"], 3),
                  "R² addestramento": np.nan, "MAE addestramento": np.nan,
                  **({"AUC": 0.5} if compito == "classificazione" else {})})
    return pd.DataFrame(righe)


# =========================================================================
# K-MEANS
# =========================================================================

def raggruppa_generico(df: pd.DataFrame, variabili: List[str], k: int, gruppo: str | None,
              per_atleta: bool, descrittiva: str | None = None, seed: int = 7) -> dict:
    if len(variabili) < 2:
        raise ValueError("Scegli almeno due variabili numeriche.")
    if per_atleta and gruppo:
        tab = df.groupby(gruppo)[variabili].mean()
        if descrittiva and descrittiva in df.columns:
            tab = df.groupby(gruppo)[[descrittiva]].agg(
                lambda s: s.mode().iat[0] if s.notna().any() else np.nan).join(tab)
        tab = tab.reset_index()
    else:
        tab = df[[c for c in ([gruppo] if gruppo else []) + ([descrittiva] if descrittiva else [])
                  + variabili if c in df.columns]].copy().reset_index(drop=True)
    X = tab[variabili].astype(float)
    X = X.fillna(X.median())
    if len(X) <= k:
        raise ValueError(f"Servono più righe ({len(X)}) che gruppi ({k}).")
    Z = StandardScaler().fit_transform(X)
    totale = float(((Z - Z.mean(axis=0)) ** 2).sum())
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    gomito = []
    for kk in range(2, min(9, len(X))):
        m = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        gomito.append({"k": kk, "R²": 1 - m.inertia_ / totale,
                       "MAE": float(np.mean(np.abs(Z - m.cluster_centers_[m.labels_])))})
    tab["gruppo"] = [f"Gruppo {l + 1}" for l in km.labels_]
    xy = PCA(n_components=2, random_state=seed).fit_transform(Z)
    tab["pca_1"], tab["pca_2"] = xy[:, 0], xy[:, 1]
    identikit = tab.groupby("gruppo")[variabili].mean().round(2)
    identikit.insert(0, "righe", tab.groupby("gruppo").size())
    comp = pd.crosstab(tab["gruppo"], tab[descrittiva]) if descrittiva and descrittiva in tab else None
    return {"compito": "raggruppamento", "modello": "K-Means", "k": k, "variabili": variabili,
            "misure": {"R² (varianza spiegata dai gruppi)": 1 - km.inertia_ / totale,
                       "MAE (distanza media dal centro, in dev. standard)":
                           float(np.mean(np.abs(Z - km.cluster_centers_[km.labels_])))},
            "gomito": pd.DataFrame(gomito), "tabella": tab, "identikit": identikit.reset_index(),
            "composizione": comp, "categoria": descrittiva}


# =========================================================================
# LETTURA DEL RISULTATO E GRAFICI SPECIFICI
# =========================================================================

def commenta_generico(ris: dict) -> List[str]:
    if ris["compito"] != "classificazione" or len(ris["classi"]) == 2:
        righe = commenta(ris)
    else:
        m = ris["misure"]
        k = len(ris["classi"])
        righe = [f"Classificazione con <b>{k} classi</b>. Accuratezza bilanciata = "
                 f"<b>{m['Accuratezza bilanciata']:.2f}</b> (tirando a caso: {1 / k:.2f}); "
                 f"AUC media = <b>{m['AUC']:.2f}</b>.",
                 f"R² sulle probabilità = <b>{m['R² (verifica)']:.2f}</b>; MAE = "
                 f"<b>{m['MAE (verifica)']:.2f}</b>: in media il modello lascia "
                 f"{m['MAE (verifica)'] * 100:.0f}% di probabilità fuori dalla classe giusta "
                 f"(il riferimento banale: {ris['riferimento']['MAE (verifica)'] * 100:.0f}%)."]
        if m["R² (addestramento)"] - m["R² (verifica)"] > 0.3:
            righe.append("⚠️ Forte distacco fra addestramento e verifica: il modello ha in parte "
                         "<b>imparato a memoria</b>.")
    if ris["compito"] == "classificazione" and len(ris["classi"]) == 2:
        righe.insert(0, f"Classe considerata «evento»: <b>{ris['classe_positiva']}</b>.")
    return righe


def g_confusione_multi(ris: dict):
    import matplotlib.pyplot as plt
    m = ris["confusione"]
    classi = ris["classi"]
    k = len(classi)
    fig, ax = plt.subplots(figsize=(1.0 * k + 3.0, 0.9 * k + 2.6))
    ax.imshow(m, cmap="BuGn")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > m.max() * 0.55 else "#22303b")
    etich = [str(c)[:14] for c in classi]
    ax.set_xticks(range(k), etich, fontsize=8, rotation=30 if k > 3 else 0)
    ax.set_yticks(range(k), etich, fontsize=8)
    ax.set_xlabel("previsto", fontsize=9)
    ax.set_ylabel("reale", fontsize=9)
    ax.set_title("Matrice di confusione", fontsize=10.5, color="#22303b")
    fig.tight_layout()
    return fig
