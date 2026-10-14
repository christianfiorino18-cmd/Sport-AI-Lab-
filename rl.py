"""
rl.py — Sport AI Lab Start · reinforcement learning

Il motore di Q-learning e i due ambienti, senza Streamlit:

    1. MOTORE          Q-learning tabellare, valutazione, lettura della politica, grafici
    2. CALCIO          l'allenatore che sceglie la tattica ogni 5 minuti   → rl.CALCIO
    3. NUTRIZIONE      il nutrizionista che sceglie i carboidrati ogni giorno → rl.NUTRIZIONE

Ogni ambiente è costruito da una funzione e restituito come un unico oggetto
(rl.CALCIO, rl.NUTRIZIONE) con le sue VARIABILI, AZIONI, RICOMPENSE, Ambiente…
così i due ambienti convivono nello stesso file senza mescolare i nomi.

Copyright (C) 2026  Christian Fiorino — licenza MIT
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ###########################################################################
# 1 · MOTORE
# ###########################################################################
# rl_motore.py — Sport AI Lab Start · motore di reinforcement learning comune
#
# Un solo algoritmo (Q-learning tabellare) per due ambienti diversi:
#
#     rl.CALCIO        l'allenatore che sceglie la tattica ogni 5 minuti
#     rl.NUTRIZIONE    il nutrizionista che sceglie i carboidrati ogni giorno
#
# Ogni ambiente dichiara le sue VARIABILI di stato (punteggio, energia, pressing
# dell'avversario… oppure tipo di giornata, riserve di glicogeno, sonno…), le
# AZIONI possibili, le RICOMPENSE (bonus e malus regolabili) e alcune politiche
# fisse di confronto. Il motore fa il resto.
#
# La novità didattica: **lo studente sceglie quali variabili l'agente può
# osservare**. Una variabile non osservata esiste ancora nella partita — il
# pressing avversario continua a cambiare le probabilità di gol — ma l'agente non
# la vede e quindi non può adattare la scelta. Confrontando addestramenti con più
# o meno variabili si capisce che cosa vuol dire «lo stato» in RL, e perché
# vedere di più costa più esperienza.
#
# Il modulo non importa Streamlit.

COLORE = {"primario": "#1b5e5a", "secondario": "#c2603a", "neutro": "#8a9199",
          "chiaro": "#a8c8c4", "scuro": "#22303b"}

MIN_VISITE = 25          # sotto questa soglia la scelta dell'agente non è affidabile


# =========================================================================
# DESCRIZIONE DI UN AMBIENTE
# =========================================================================

@dataclass
class Variabile:
    nome: str
    etichetta: str
    livelli: List[str]
    descrizione: str


@dataclass
class Ricompensa:
    chiave: str
    etichetta: str
    valore: float
    minimo: float
    massimo: float
    passo: float
    aiuto: str = ""


@dataclass
class Contesto:
    chiave: str
    etichetta: str
    valore: float
    minimo: float
    massimo: float
    passo: float
    aiuto: str = ""


@dataclass
class Impostazioni:
    episodi: int = 20000
    alfa: float = 0.01           # passo minimo di apprendimento
    gamma: float = 0.95          # peso del futuro
    epsilon: float = 0.30        # esplorazione iniziale
    seed: int = 7


# =========================================================================
# APPRENDIMENTO
# =========================================================================

def _proiettore(D, osservate: Sequence[str]):
    """Riduce lo stato completo alle sole variabili osservate (le altre → 0)."""
    maschera = [v.nome in osservate for v in D.VARIABILI]
    forma = tuple(len(v.livelli) if m else 1 for v, m in zip(D.VARIABILI, maschera))

    def proietta(s):
        return tuple(x if m else 0 for x, m in zip(s, maschera))
    return proietta, forma


def allena(D, ric: Dict[str, float], contesto: Dict[str, float], imp: Impostazioni,
           osservate: Sequence[str]) -> dict:
    """
    Q-learning tabellare. Il passo di apprendimento parte alto e si riduce con
    il numero di visite di ogni (stato, azione), fino al minimo `alfa`: così gli
    stati rari imparano in fretta e quelli frequenti non rincorrono il rumore.
    """
    osservate = [v.nome for v in D.VARIABILI if v.nome in osservate]
    rng = random.Random(imp.seed)          # generatore veloce, per milioni di estrazioni
    env = D.Ambiente(ric, contesto, rng)
    proietta, forma = _proiettore(D, osservate)
    nA = len(D.AZIONI)
    azioni = range(nA)
    # tabella Q come dizionario di liste: molto più veloce di numpy cella per cella
    Qd: Dict[tuple, list] = {}
    Nd: Dict[tuple, list] = {}
    storia = []
    for ep in range(imp.episodi):
        s = proietta(env.reset())
        eps = max(0.02, imp.epsilon * (1 - ep / imp.episodi))
        totale = 0.0
        while True:
            q = Qd.get(s)
            if q is None:
                q = Qd[s] = [0.0] * nA
                Nd[s] = [0] * nA
            if rng.random() < eps:
                a = rng.randrange(nA)
            else:
                m = max(q)
                a = q.index(m)
            s2_pieno, r, fine = env.passo(a)
            s2 = proietta(s2_pieno)
            if fine:
                obiettivo = r
            else:
                q2 = Qd.get(s2)
                obiettivo = r + imp.gamma * (max(q2) if q2 is not None else 0.0)
            n = Nd[s]
            n[a] += 1
            alfa = max(imp.alfa, 1.0 / n[a])
            q[a] += alfa * (obiettivo - q[a])
            totale += r
            s = s2
            if fine:
                break
        storia.append({"episodio": ep + 1, "ricompensa": totale, **env.esito()})
    Q = np.zeros(forma + (nA,))
    N = np.zeros(forma + (nA,), dtype=np.int32)
    for k, q in Qd.items():
        Q[k] = q
        N[k] = Nd[k]
    visite = N.sum(axis=-1)
    return {"Q": Q, "visite": visite, "politica": np.argmax(Q, axis=-1),
            "osservate": osservate, "forma": forma,
            "n_stati": int(np.prod([len(v.livelli) for v in D.VARIABILI if v.nome in osservate]))
            if osservate else 1,
            "storia": pd.DataFrame(storia), "ricompense": dict(ric),
            "contesto": dict(contesto), "impostazioni": asdict(imp)}


def scelta_agente(D, ris: dict) -> Callable:
    proietta, _ = _proiettore(D, ris["osservate"])
    pol = ris["politica"]
    return lambda s: int(pol[proietta(s)])


# =========================================================================
# VALUTAZIONE
# =========================================================================

def gioca(D, scegli: Callable, ric, contesto, seed: int) -> pd.DataFrame:
    """Un episodio con il diario passo per passo (il 'dataset' dell'agente)."""
    rng = random.Random(seed)
    env = D.Ambiente(ric, contesto, rng)
    s = env.reset()
    righe = []
    while True:
        prima = env.leggi()
        a = int(scegli(s))
        s, r, fine = env.passo(a)
        righe.append({**prima, D.NOME_AZIONE: D.AZIONI[a], **env.ultimo, "ricompensa": round(r, 2)})
        if fine:
            break
    return pd.DataFrame(righe)


def valuta(D, scegli: Callable, ric, contesto, episodi: int = 200, seme: int = 10_000) -> dict:
    """Media su molti episodi identici per tutti (stessi semi)."""
    tot, esiti = [], []
    for i in range(episodi):
        env = D.Ambiente(ric, contesto, random.Random(seme + i))
        s = env.reset()
        somma = 0.0
        while True:
            s, r, fine = env.passo(int(scegli(s)))
            somma += r
            if fine:
                break
        tot.append(somma)
        esiti.append(env.esito())
    medie = pd.DataFrame(esiti).mean().round(2).to_dict()
    return {"ricompensa media": round(float(np.mean(tot)), 1), **medie}


def confronto(D, ris: dict, ric, contesto, episodi: int = 200) -> pd.DataFrame:
    righe = [{"strategia": "🤖 agente Q-learning",
              **valuta(D, scelta_agente(D, ris), ric, contesto, episodi)}]
    for nome, regola in D.POLITICHE_FISSE.items():
        righe.append({"strategia": nome, **valuta(D, regola, ric, contesto, episodi)})
    return (pd.DataFrame(righe).sort_values("ricompensa media", ascending=False)
            .reset_index(drop=True))


def precisione_q(D, ris: dict, ric, contesto, episodi: int = 120) -> dict:
    """R² e MAE fra il valore Q (ricompensa futura prevista) e quella ottenuta."""
    proietta, _ = _proiettore(D, ris["osservate"])
    Q, pol, gamma = ris["Q"], ris["politica"], ris["impostazioni"]["gamma"]
    stimati, reali = [], []
    for i in range(episodi):
        env = D.Ambiente(ric, contesto, random.Random(30_000 + i))
        s = proietta(env.reset())
        qs, rs = [], []
        while True:
            a = int(pol[s])
            qs.append(float(Q[s + (a,)]))
            s2, r, fine = env.passo(a)
            rs.append(r)
            s = proietta(s2)
            if fine:
                break
        G, rit = 0.0, []
        for r in reversed(rs):
            G = r + gamma * G
            rit.append(G)
        stimati += qs
        reali += rit[::-1]
    y, p = np.array(reali), np.array(stimati)
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"R²": r2, "MAE": float(np.mean(np.abs(y - p)))}


# =========================================================================
# LETTURA DELLA POLITICA
# =========================================================================

def _stati_visti(D, ris: dict):
    """Tutti gli stati (nelle sole variabili osservate) con abbastanza visite."""
    idx = [i for i, v in enumerate(D.VARIABILI) if v.nome in ris["osservate"]]
    for combo in product(*[range(len(D.VARIABILI[i].livelli)) for i in idx]):
        s = [0] * len(D.VARIABILI)
        for i, x in zip(idx, combo):
            s[i] = x
        s = tuple(s)
        if ris["visite"][s] >= MIN_VISITE:
            yield s


def tabella_politica(D, ris: dict) -> pd.DataFrame:
    righe = []
    for s in _stati_visti(D, ris):
        riga = {v.etichetta: v.livelli[s[i]] for i, v in enumerate(D.VARIABILI)
                if v.nome in ris["osservate"]}
        riga[D.NOME_AZIONE] = D.AZIONI[int(ris["politica"][s])]
        riga["volte incontrata"] = int(ris["visite"][s])
        righe.append(riga)
    return pd.DataFrame(righe)


def tabella_q(D, ris: dict) -> pd.DataFrame:
    righe = []
    for s in _stati_visti(D, ris):
        base = {v.etichetta: v.livelli[s[i]] for i, v in enumerate(D.VARIABILI)
                if v.nome in ris["osservate"]}
        for a, nome in enumerate(D.AZIONI):
            righe.append({**base, D.NOME_AZIONE: nome,
                          "valore_Q": round(float(ris["Q"][s + (a,)]), 3)})
    return pd.DataFrame(righe)


def importanza_variabili(D, ris: dict, margine: float = 1.0) -> pd.DataFrame:
    """
    Quanto ogni variabile cambia la decisione dell'agente: per ogni stato visto
    si cambia il livello di UNA variabile, lasciando le altre ferme, e si conta
    quante volte la scelta cambia in modo rilevante (la scelta dell'altro
    livello costerebbe almeno `margine` punti). È l'equivalente RL dell'importanza delle
    variabili di un Random Forest.
    """
    visti = set(_stati_visti(D, ris))
    righe = []
    for i, v in enumerate(D.VARIABILI):
        if v.nome not in ris["osservate"]:
            continue
        confronti = cambi = 0
        for s in visti:
            for liv in range(len(v.livelli)):
                if liv == s[i]:
                    continue
                t = s[:i] + (liv,) + s[i + 1:]
                if t in visti:
                    confronti += 1
                    # conta solo i cambi che contano: usare in s la scelta fatta in t
                    # costerebbe almeno `margine` punti di ricompensa attesa
                    a_s, a_t = ris["politica"][s], ris["politica"][t]
                    cambi += int(ris["Q"][s + (a_s,)] - ris["Q"][s + (a_t,)] >= margine)
        righe.append({"variabile": v.etichetta,
                      "decisioni che cambia (%)": round(100 * cambi / confronti, 1)
                      if confronti else 0.0})
    return (pd.DataFrame(righe).sort_values("decisioni che cambia (%)", ascending=False)
            .reset_index(drop=True))


def uso_per_variabile(D, ris: dict, ric, contesto, nome_var: str,
                      episodi: int = 120) -> pd.DataFrame:
    """Quanto spesso l'agente usa ciascuna azione, per livello di una variabile (%)."""
    scegli = scelta_agente(D, ris)
    v = next(x for x in D.VARIABILI if x.nome == nome_var)
    i = D.VARIABILI.index(v)
    conte = np.zeros((len(v.livelli), len(D.AZIONI)))
    for k in range(episodi):
        env = D.Ambiente(ric, contesto, random.Random(20_000 + k))
        s = env.reset()
        while True:
            a = scegli(s)
            conte[s[i], a] += 1
            s, _, fine = env.passo(a)
            if fine:
                break
    tot = conte.sum(axis=1, keepdims=True)
    pct = np.where(tot > 0, conte / np.maximum(tot, 1) * 100, np.nan)
    t = pd.DataFrame(pct.round(0), index=v.livelli, columns=D.AZIONI)
    t.index.name = v.etichetta
    t["situazioni incontrate"] = tot.ravel().astype(int)
    return t


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


def g_apprendimento(D, ris: dict) -> plt.Figure:
    st = ris["storia"]
    finestra = max(50, len(st) // 25)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(st["episodio"], st["ricompensa"].rolling(finestra, min_periods=10).mean(),
            color=COLORE["primario"], linewidth=2.4,
            label=f"ricompensa media (ultimi {finestra} episodi)")
    _asse(ax, "L'agente migliora provando", "episodi giocati", "ricompensa media")
    col, etich = D.SERIE_SECONDARIA
    if col in st.columns:
        ax2 = ax.twinx()
        ax2.plot(st["episodio"], st[col].rolling(finestra, min_periods=10).mean(),
                 color=COLORE["secondario"], linewidth=1.4, alpha=0.85, label=etich)
        ax2.set_ylabel(etich, fontsize=9, color=COLORE["secondario"])
        ax2.tick_params(labelsize=8.5, colors=COLORE["secondario"])
        ax2.spines["top"].set_visible(False)
        linee = ax.get_lines() + ax2.get_lines()
        ax.legend(linee, [l.get_label() for l in linee], fontsize=8, frameon=False,
                  loc="lower right")
    fig.tight_layout()
    return fig


def g_confronto(tab: pd.DataFrame) -> plt.Figure:
    d = tab.iloc[::-1]
    colori = [COLORE["primario"] if "agente" in s else COLORE["neutro"] for s in d["strategia"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.4))
    ax.barh(d["strategia"], d["ricompensa media"], color=colori)
    ax.axvline(0, color=COLORE["scuro"], linewidth=0.8)
    _asse(ax, "Ricompensa media per episodio", "ricompensa")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def g_importanza(imp: pd.DataFrame) -> plt.Figure:
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(d) + 1.4))
    ax.barh(d["variabile"], d["decisioni che cambia (%)"], color=COLORE["primario"])
    ax.set_xlim(0, 100)
    _asse(ax, "Quanto ogni variabile cambia la decisione dell'agente",
          "% di decisioni che cambiano se cambia solo questa variabile")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def g_mappa(D, ris: dict, var_righe: str, var_colonne: str, fissi: Dict[str, int]) -> plt.Figure:
    """Mappa delle decisioni su due variabili, tenendo fisse le altre."""
    nomi = [v.nome for v in D.VARIABILI]
    ir, ic = nomi.index(var_righe), nomi.index(var_colonne)
    vr, vc = D.VARIABILI[ir], D.VARIABILI[ic]
    M = np.full((len(vr.livelli), len(vc.livelli)), -1)
    for r in range(len(vr.livelli)):
        for c in range(len(vc.livelli)):
            s = [0] * len(nomi)
            for n, liv in fissi.items():
                if n in ris["osservate"]:
                    s[nomi.index(n)] = liv
            s[ir], s[ic] = r, c
            s = tuple(s)
            if ris["visite"][s] >= MIN_VISITE:
                M[r, c] = int(ris["politica"][s])
    cmap = plt.get_cmap("Set2", len(D.AZIONI))
    fig, ax = plt.subplots(figsize=(1.35 * len(vc.livelli) + 2.6, 0.62 * len(vr.livelli) + 1.6))
    ax.imshow(np.ones_like(M), cmap="Greys", vmin=0, vmax=8, aspect="auto")
    ax.imshow(np.where(M < 0, np.nan, M), cmap=cmap, vmin=-0.5, vmax=len(D.AZIONI) - 0.5,
              aspect="auto")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, "—" if M[r, c] < 0 else D.SIGLE[M[r, c]], ha="center", va="center",
                    fontsize=8.5, color=COLORE["neutro"] if M[r, c] < 0 else COLORE["scuro"])
    ax.set_xticks(range(len(vc.livelli)), vc.livelli, fontsize=8)
    ax.set_yticks(range(len(vr.livelli)), vr.livelli, fontsize=8)
    ax.set_xlabel(vc.etichetta, fontsize=9)
    ax.set_ylabel(vr.etichetta, fontsize=9)
    ax.set_title("Che cosa sceglie l'agente", fontsize=10.5, color=COLORE["scuro"])
    fig.text(0.5, 0.01, "  ·  ".join(f"{s} = {a}" for s, a in zip(D.SIGLE, D.AZIONI))
             + "  ·  — = quasi mai incontrata", ha="center", fontsize=7.5, color=COLORE["scuro"],
             wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ###########################################################################
# 2 · AMBIENTE CALCIO
# ###########################################################################
# rl_calcio.py — Sport AI Lab Start · ambiente RL «tattiche di gioco»
#
# Una partita di 90 minuti divisa in 18 blocchi da 5 minuti. In ogni blocco
# l'allenatore (l'agente) sceglie una tattica. Le variabili in gioco sono sei,
# e lo studente decide quali l'agente può vedere:
#
#     punteggio         sotto · pari · sopra
#     energia           bassa · media · alta — cala pressando, risale all'intervallo
#     fase              1'–22' · 23'–45' · 46'–67' · 68'–90'
#     avversario        attendista · equilibrato · pressing alto — l'avversario cambia
#                       atteggiamento durante la partita, soprattutto quando perde
#     gol subito        no · sì, negli ultimi 10 minuti (la squadra è scossa)
#     uomini in campo   10 contro 11 · 11 contro 11 · 11 contro 10 (espulsioni)
#
# Ogni variabile cambia davvero le probabilità di gol: contro un avversario che
# alza il pressing il possesso diventa rischioso e il contropiede trova spazio;
# contro un avversario chiuso il contropiede non serve; appena subito un gol
# sbilanciarsi costa caro; in dieci uomini il pressing è un suicidio.

def _poisson(rng, lam: float) -> int:
    """Estrazione di Poisson (algoritmo di Knuth): veloce per medie piccole."""
    soglia, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= soglia:
            return k
        k += 1


def _ambiente_calcio() -> SimpleNamespace:
    """Costruisce l'ambiente e lo restituisce come un unico oggetto."""
    CHIAVE = "calcio"
    TITOLO = "⚽ RL Calcio · tattiche di gioco"
    UNITA = "blocco da 5 minuti"
    EPISODIO = "partita"
    NOME_AZIONE = "tattica"

    AZIONI = ["Pressing alto", "Blocco medio", "Difesa bassa", "Contropiede", "Possesso palla"]
    SIGLE = ["PRESS", "MEDIO", "BASSA", "CONTR", "POSS"]
    P, M, B, C, POS = range(5)

    VARIABILI = [
        Variabile("punteggio", "punteggio", ["sotto", "pari", "sopra"],
                  "Stiamo perdendo, pareggiando o vincendo."),
        Variabile("energia", "energia", ["bassa", "media", "alta"],
                  "Riserva fisica della squadra: il pressing la consuma, l'intervallo la ricarica."),
        Variabile("fase", "fase", ["1'–22'", "23'–45'", "46'–67'", "68'–90'"],
                  "Il momento della partita: un gol al 10' e uno all'85' non pesano allo stesso modo."),
        Variabile("avversario", "atteggiamento avversario", ["attendista", "equilibrato", "pressing alto"],
                  "Cosa fa l'avversario. Cambia durante la gara: se perde tende ad alzare il pressing, "
                  "se vince a chiudersi."),
        Variabile("scossa", "gol subito da poco", ["no", "sì"],
                  "Abbiamo subito un gol negli ultimi 10 minuti: la squadra è nervosa e si sbilancia."),
        Variabile("uomini", "uomini in campo", ["10 vs 11", "11 vs 11", "11 vs 10"],
                  "Inferiorità o superiorità numerica dopo un'espulsione."),
    ]
    OSSERVATE_INIZIALI = ["punteggio", "energia", "fase", "avversario", "scossa", "uomini"]

    RICOMPENSE = [
        Ricompensa("gol_fatto", "⚽ Gol fatto", 10, -50, 50, 1),
        Ricompensa("gol_subito", "🥅 Gol subito", -10, -50, 50, 1),
        Ricompensa("palla_recuperata", "🔄 Palla recuperata", 0.5, -5, 5, 0.1),
        Ricompensa("energia_esaurita", "🪫 Squadra senza energia", -2, -20, 5, 0.5,
                   "Per ogni blocco giocato con energia sotto 0,25."),
        Ricompensa("espulsione", "🟥 Nostro giocatore espulso", -8, -40, 5, 1,
                   "Il pressing a squadra stanca e il nervosismo dopo un gol subito aumentano i falli."),
        Ricompensa("vittoria", "🏆 Vittoria (fine partita)", 15, -50, 80, 1),
        Ricompensa("pareggio", "🤝 Pareggio (fine partita)", 3, -50, 50, 1),
        Ricompensa("sconfitta", "❌ Sconfitta (fine partita)", -15, -80, 50, 1),
    ]
    PRESET = {
        "⚖️ Equilibrato": {},
        "🧱 Catenaccio": {"gol_subito": -30, "sconfitta": -30, "pareggio": 8, "palla_recuperata": 0.2},
        "🔥 Spettacolo": {"gol_fatto": 25, "gol_subito": -8, "palla_recuperata": 1.5,
                         "energia_esaurita": 0},
        "🏆 Solo il risultato": {"gol_fatto": 0, "gol_subito": 0, "palla_recuperata": 0,
                                "energia_esaurita": 0, "espulsione": 0, "vittoria": 30,
                                "pareggio": 5, "sconfitta": -30},
    }
    CONTESTO = [
        Contesto("forza", "Forza dell'avversario", 1.0, 0.7, 1.3, 0.05,
                 "0,7 = molto più debole · 1,0 = pari livello · 1,3 = molto più forte"),
        Contesto("imprevedibile", "Quanto spesso l'avversario cambia atteggiamento", 0.12, 0.0, 0.4, 0.02,
                 "Probabilità, ogni 5 minuti, che l'avversario cambi tattica per conto suo."),
    ]
    TURNI = 18
    SERIE_SECONDARIA = ("vittoria", "% vittorie")

    # profilo di base di ogni tattica, per blocco di 5', contro un avversario equilibrato
    #           gol fatto  gol subito  palle recuperate  energia spesa
    EFFETTI = {P: (0.070, 0.040, 2.6, 0.085),
               M: (0.042, 0.034, 1.6, 0.040),
               B: (0.018, 0.026, 1.0, 0.020),
               C: (0.040, 0.030, 1.3, 0.045),
               POS: (0.048, 0.022, 0.8, 0.025)}




    class Ambiente:
        def __init__(self, ric: dict, contesto: dict, rng: np.random.Generator):
            self.ric, self.ctx, self.rng = ric, contesto, rng
            self.ultimo = {}

        # --------------------------------------------------------------- stato
        def reset(self):
            self.blocco = 0
            self.gf = self.gs = 0
            self.energia = 1.0
            self.avv = self.rng.choices((0, 1, 2), (0.25, 0.5, 0.25))[0]
            self.ultimo_gol_subito = -99
            self.noi, self.loro = 11, 11
            self.recuperi = self.espulsioni = 0
            self.ultimo = {}
            return self.stato()

        def stato(self):
            d = self.gf - self.gs
            return (0 if d < 0 else (1 if d == 0 else 2),
                    0 if self.energia < 0.35 else (1 if self.energia < 0.65 else 2),
                    min(3, self.blocco * 4 // TURNI),
                    self.avv,
                    int(self.blocco - self.ultimo_gol_subito <= 2),
                    0 if self.noi < self.loro else (1 if self.noi == self.loro else 2))

        def leggi(self) -> dict:
            s = self.stato()
            return {"minuto": (self.blocco + 1) * 5, "risultato": f"{self.gf}-{self.gs}",
                    **{v.etichetta: v.livelli[x] for v, x in zip(VARIABILI, s)},
                    "energia (0–1)": round(self.energia, 2)}

        # --------------------------------------------------------------- passo
        def passo(self, a: int):
            gf0, gs0, ec, costo = EFFETTI[a]
            forza = self.ctx.get("forza", 1.0)
            d = self.gf - self.gs
            stanco = max(0.0, 0.5 - self.energia) * 2                   # 0 … 1
            scossa = self.blocco - self.ultimo_gol_subito <= 2
            p_gf = gf0 / forza * (1 - 0.5 * stanco)
            p_gs = gs0 * forza * (1 + 1.2 * stanco)

            if a == P:
                p_gs *= 1 + 1.5 * stanco                                 # pressing stanco = spazi
            if a == C and d > 0:
                p_gf *= 1.5
            if a == POS and d > 0:
                p_gs *= 0.8
            if d < 0 and a in (M, B):
                p_gf *= 0.8

            # l'avversario
            if self.avv == 2:                                            # alza il pressing
                if a == POS:
                    p_gs *= 1.9; p_gf *= 0.8                              # palle perse in uscita
                elif a == C:
                    p_gf *= 1.9                                          # spazio alle sue spalle
                elif a == M:
                    p_gs *= 1.2
                elif a == P:
                    p_gs *= 1.25; p_gf *= 1.1                            # partita a viso aperto
                costo += 0.010                                           # si corre di più
            elif self.avv == 0:                                          # si chiude
                if a == C:
                    p_gf *= 0.45                                         # nessuno spazio
                elif a in (POS, P):
                    p_gf *= 1.25
                elif a == B:
                    p_gf *= 0.6
                p_gs *= 0.7

            # gol appena subito: sbilanciarsi costa caro, ritrovare ordine aiuta
            if scossa:
                if a in (P, C):
                    p_gs *= 1.6
                elif a in (M, POS):
                    p_gs *= 0.9

            # uomini in campo
            if self.noi < self.loro:
                p_gf *= 0.6; p_gs *= 1.6
                if a == P:
                    p_gs *= 1.4; costo *= 1.3
                if a == B:
                    p_gs *= 0.8
            elif self.noi > self.loro:
                p_gf *= 1.5; p_gs *= 0.7
                if a in (POS, P):
                    p_gf *= 1.2

            gol_f = int(self.rng.random() < p_gf)
            gol_s = int(self.rng.random() < p_gs)
            rec = _poisson(self.rng, ec * (1 - 0.4 * stanco))
            p_rosso = 0.004 * (3.0 if (a == P and stanco > 0.3) else 1.0) * (1.8 if scossa else 1.0)
            rosso_noi = int(self.noi > 9 and self.rng.random() < p_rosso)
            rosso_loro = int(self.loro > 9 and self.rng.random() < 0.004 * (2.0 if a == POS else 1.0))

            self.gf += gol_f
            self.gs += gol_s
            self.noi -= rosso_noi
            self.loro -= rosso_loro
            self.recuperi += rec
            self.espulsioni += rosso_noi
            if gol_s:
                self.ultimo_gol_subito = self.blocco + 1
            self.energia = min(1.0, max(0.0, self.energia - costo + 0.015 + self.rng.gauss(0, 0.01)))
            self.blocco += 1
            if self.blocco == TURNI // 2:
                self.energia = min(1.0, self.energia + 0.25)              # intervallo

            # l'avversario cambia atteggiamento
            d = self.gf - self.gs
            if d > 0 and self.rng.random() < 0.30:
                self.avv = 2                                             # perde: attacca
            elif d < 0 and self.rng.random() < 0.25:
                self.avv = 0                                             # vince: si chiude
            elif self.rng.random() < self.ctx.get("imprevedibile", 0.12):
                self.avv = self.rng.randrange(3)

            r = self.ric
            premio = (gol_f * r["gol_fatto"] + gol_s * r["gol_subito"] + rec * r["palla_recuperata"]
                      + rosso_noi * r["espulsione"])
            if self.energia < 0.25:
                premio += r["energia_esaurita"]
            fine = self.blocco >= TURNI
            esito = ""
            if fine:
                esito = "vittoria" if self.gf > self.gs else ("pareggio" if self.gf == self.gs
                                                              else "sconfitta")
                premio += r[esito]
            eventi = []
            if gol_f: eventi.append("⚽ gol fatto")
            if gol_s: eventi.append("🥅 gol subito")
            if rosso_noi: eventi.append("🟥 nostro espulso")
            if rosso_loro: eventi.append("🟥 loro espulso")
            self.ultimo = {"gol_fatto": gol_f, "gol_subito": gol_s, "palle_recuperate": rec,
                           "eventi": ", ".join(eventi), "esito": esito}
            return self.stato(), float(premio), fine

        def esito(self) -> dict:
            return {"gol fatti": self.gf, "gol subiti": self.gs,
                    "vittoria": 100.0 * (self.gf > self.gs), "pareggio": 100.0 * (self.gf == self.gs),
                    "sconfitta": 100.0 * (self.gf < self.gs), "espulsioni": self.espulsioni}


    def _regola_allenatore(s):
        """Una regola «da manuale» scritta a mano, che guarda tutte le variabili."""
        punteggio, energia, fase, avv, scossa, uomini = s
        if uomini == 0:
            return B
        if scossa:
            return POS
        if avv == 2:
            return C
        if punteggio == 2 and fase == 3:
            return POS
        if punteggio == 0 and fase >= 2:
            return P if energia >= 1 else POS
        return P if energia == 2 else M


    POLITICHE_FISSE = {**{f"sempre {t}": (lambda s, i=i: i) for i, t in enumerate(AZIONI)},
                       "📋 regola dell'allenatore (scritta a mano)": _regola_allenatore}

    COLONNE_CONFRONTO = ["strategia", "ricompensa media", "gol fatti", "gol subiti", "vittoria",
                         "pareggio", "sconfitta", "espulsioni"]
    RINOMINA_CONFRONTO = {"vittoria": "vittorie %", "pareggio": "pareggi %", "sconfitta": "sconfitte %"}


    def metriche_episodio(log) -> dict:
        gf, gs = int(log["gol_fatto"].sum()), int(log["gol_subito"].sum())
        return {"Risultato": f"{gf} – {gs}", "Esito": log["esito"].iloc[-1],
                "Palle recuperate": int(log["palle_recuperate"].sum()),
                "Ricompensa totale": f"{log['ricompensa'].sum():.1f}"}


    def g_episodio(log):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(log["minuto"], log["energia (0–1)"], color="#1b5e5a", linewidth=2, label="energia")
        livelli = {"attendista": 0.05, "equilibrato": 0.10, "pressing alto": 0.15}
        ax.scatter(log["minuto"], [livelli[x] for x in log["atteggiamento avversario"]],
                   s=14, color="#8a9199", label="avversario (basso → alto pressing)")
        for _, r in log.iterrows():
            if r["gol_fatto"]:
                ax.axvline(r["minuto"], color="#1b5e5a", linestyle="--", alpha=0.7)
                ax.text(r["minuto"], 1.0, "gol +", fontsize=8, ha="center")
            if r["gol_subito"]:
                ax.axvline(r["minuto"], color="#c2603a", linestyle="--", alpha=0.7)
                ax.text(r["minuto"], 0.92, "gol −", fontsize=8, ha="center", color="#c2603a")
        ax.set_ylim(0, 1.08)
        ax.set_title("La partita simulata", fontsize=10.5, color="#22303b")
        ax.set_xlabel("minuto", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)
        fig.tight_layout()
        return fig

    IMPOSTAZIONI = {"episodi": 40000, "gamma": 0.95}
    EPISODI_SCELTE = [5000, 20000, 40000, 80000]

    return SimpleNamespace(**locals())


CALCIO = _ambiente_calcio()


# ###########################################################################
# 3 · AMBIENTE NUTRIZIONE
# ###########################################################################
# rl_nutrizione.py — Sport AI Lab Start · ambiente RL «periodizzazione dei carboidrati»
#
# Quattro settimane di preparazione di un atleta di endurance (70 kg). Ogni giorno
# il nutrizionista (l'agente) sceglie quanti carboidrati prescrivere:
#
#     3 · 5 · 7 · 9 · 11 g/kg
#
# Le variabili in gioco sono cinque, e lo studente decide quali l'agente può vedere:
#
#     giornata      riposo · leggero · intenso · gara
#     domani        allenamento · gara (la gara di domani si prepara oggi)
#     riserve       glicogeno basso · medio · pieno
#     bilancio      equilibrio · deficit · surplus (energia accumulata nei giorni)
#     sonno         buono · scarso (dormire male rallenta il recupero del glicogeno
#                   e rende l'intestino più sensibile)
#
# Bonus e malus li decide lo studente: qualità della seduta, gara ben riuscita,
# giornate in bassa disponibilità energetica, disturbi gastrointestinali,
# aumento di peso indesiderato.

def _ambiente_nutrizione() -> SimpleNamespace:
    """Costruisce l'ambiente e lo restituisce come un unico oggetto."""
    CHIAVE = "nutri"
    TITOLO = "🍝 RL Nutrizione · periodizzazione dei carboidrati"
    UNITA = "giorno"
    EPISODIO = "mese di preparazione"
    NOME_AZIONE = "carboidrati"

    AZIONI = ["3 g/kg", "5 g/kg", "7 g/kg", "9 g/kg", "11 g/kg"]
    SIGLE = ["3", "5", "7", "9", "11"]
    CHO = [3.0, 5.0, 7.0, 9.0, 11.0]
    MASSA = 70.0

    TIPI = ["riposo", "leggero", "intenso", "gara"]
    FABBISOGNO = {"riposo": 3.0, "leggero": 5.0, "intenso": 7.0, "gara": 8.5}
    DISPENDIO = {"riposo": 150.0, "leggero": 600.0, "intenso": 1100.0, "gara": 1500.0}
    PESO_SEDUTA = {"riposo": 0.3, "leggero": 0.8, "intenso": 1.5, "gara": 2.5}

    VARIABILI = [
        Variabile("giornata", "tipo di giornata", TIPI,
                  "Cosa prevede il programma di oggi: più è dura, più carboidrati servono."),
        Variabile("domani", "domani", ["allenamento", "gara"],
                  "Se domani c'è gara, i carboidrati di oggi riempiono le riserve (carico pre-gara)."),
        Variabile("riserve", "riserve di glicogeno", ["basse", "medie", "piene"],
                  "Glicogeno disponibile stamattina: decide la qualità della seduta."),
        Variabile("bilancio", "bilancio energetico", ["equilibrio", "deficit", "surplus"],
                  "Energia accumulata negli ultimi giorni: in deficit si rischia la bassa "
                  "disponibilità energetica, in surplus si prende peso."),
        Variabile("sonno", "sonno della notte", ["buono", "scarso"],
                  "Una notte corta rallenta la risintesi del glicogeno e rende l'intestino più sensibile."),
    ]
    OSSERVATE_INIZIALI = ["giornata", "domani", "riserve", "bilancio", "sonno"]

    RICOMPENSE = [
        Ricompensa("qualita", "🏃 Qualità della seduta (per punto)", 1.0, 0, 5, 0.1,
                   "Moltiplica la qualità della seduta, che dipende dalle riserve e dal tipo di giornata."),
        Ricompensa("gara_riuscita", "🏅 Gara con riserve piene", 10, -20, 50, 1),
        Ricompensa("bassa_ea", "⚠️ Giornata in bassa disponibilità energetica", -6, -30, 5, 0.5),
        Ricompensa("disturbo_gi", "🤢 Disturbo gastrointestinale", -4, -30, 5, 0.5,
                   "Probabile quando si danno molti più carboidrati del necessario, peggio se si è "
                   "dormito male."),
        Ricompensa("aumento_peso", "⚖️ Giornata in surplus (aumento di peso)", -2, -20, 5, 0.5),
    ]
    PRESET = {
        "⚖️ Equilibrato": {},
        "🏅 Tutto per la gara": {"gara_riuscita": 30, "qualita": 0.5, "disturbo_gi": -2},
        "🛡️ Salute prima di tutto": {"bassa_ea": -20, "disturbo_gi": -10, "gara_riuscita": 5},
        "⚖️ Controllo del peso": {"aumento_peso": -8, "bassa_ea": -3},
    }
    CONTESTO = [
        Contesto("volume", "Volume di allenamento", 1.0, 0.8, 1.2, 0.05,
                 "Moltiplica fabbisogno e dispendio: 1,2 = settimane di carico."),
        Contesto("intestino", "Sensibilità intestinale dell'atleta", 1.0, 0.5, 1.5, 0.1,
                 "1,5 = intestino delicato: i carichi alti causano più disturbi."),
    ]
    TURNI = 28
    SERIE_SECONDARIA = ("gare riuscite", "% gare con riserve piene")


    class Ambiente:
        def __init__(self, ric: dict, contesto: dict, rng):
            self.ric, self.ctx, self.rng = ric, contesto, rng
            self.ultimo = {}

        def _programma(self):
            """Calendario di 4 settimane: riposo il lunedì, gara la domenica (spesso)."""
            r = self.rng
            p = []
            for g in range(TURNI + 2):
                gs = g % 7
                if gs == 0:
                    p.append("riposo")
                elif gs == 6:
                    p.append("gara" if r.random() < 0.65 else "leggero")
                elif gs == 5:
                    p.append("leggero" if r.random() < 0.7 else "riposo")   # scarico pre-gara
                else:
                    p.append("intenso" if r.random() < 0.55 else "leggero")
            return p

        def reset(self):
            self.giorno = 0
            self.prog = self._programma()
            self.riserva = 0.7
            self.saldo = 0.0                     # kcal accumulate, con memoria che svanisce
            self.sonno_scarso = int(self.rng.random() < 0.3)
            self.gare = self.gare_ok = 0
            self.lea = self.gi = self.surplus = 0
            self.qualita_tot = 0.0
            self.cho_tot = 0.0
            self.ultimo = {}
            return self.stato()

        def stato(self):
            t = TIPI.index(self.prog[self.giorno])
            domani = int(self.prog[self.giorno + 1] == "gara")
            r = 0 if self.riserva < 0.4 else (1 if self.riserva < 0.72 else 2)
            b = 1 if self.saldo < -1500 else (2 if self.saldo > 1800 else 0)
            return (t, domani, r, b, self.sonno_scarso)

        def leggi(self) -> dict:
            s = self.stato()
            return {"giorno": self.giorno + 1,
                    **{v.etichetta: v.livelli[x] for v, x in zip(VARIABILI, s)},
                    "glicogeno (0–1)": round(self.riserva, 2), "saldo (kcal)": round(self.saldo)}

        def passo(self, a: int):
            r, ric = self.rng, self.ric
            vol = self.ctx.get("volume", 1.0)
            tipo = self.prog[self.giorno]
            domani_gara = self.prog[self.giorno + 1] == "gara"
            cho = CHO[a]
            serve = FABBISOGNO[tipo] * vol
            scarso = self.sonno_scarso

            # 1. la seduta di oggi dipende dalle riserve di stamattina
            qualita = PESO_SEDUTA[tipo] * (6.0 * self.riserva - 1.0) * (0.85 if scarso else 1.0)
            premio = ric["qualita"] * qualita
            gara_ok = 0
            if tipo == "gara":
                self.gare += 1
                if self.riserva >= 0.72:
                    gara_ok = 1
                    self.gare_ok += 1
                    premio += ric["gara_riuscita"]

            # 2. le riserve seguono lo scarto fra prescritto e necessario
            effetto = 0.08 * (cho - serve)
            if scarso and effetto > 0:
                effetto *= 0.65                                      # risintesi più lenta
            if domani_gara and cho >= 9:
                effetto += 0.10                                      # carico pre-gara
            self.riserva = min(1.0, max(0.0, self.riserva + effetto + r.gauss(0, 0.04)))

            # 3. bilancio energetico (4 kcal per grammo di carboidrati, il resto fisso)
            apporto = cho * MASSA * 4 + 1000
            spesa = 1750 + DISPENDIO[tipo] * vol
            self.saldo = 0.8 * self.saldo + (apporto - spesa)
            eventi = []
            if self.saldo < -1500:
                self.lea += 1
                premio += ric["bassa_ea"]
                eventi.append("⚠️ bassa disponibilità energetica")
            if self.saldo > 1800:
                self.surplus += 1
                premio += ric["aumento_peso"]
                eventi.append("⚖️ surplus")

            # 4. disturbi gastrointestinali: troppi carboidrati rispetto al bisogno
            eccesso = cho - serve
            p_gi = 0.0
            if eccesso >= 2.5:
                p_gi = 0.25 + 0.08 * (eccesso - 2.5)
            if cho >= 11:
                p_gi += 0.12
            p_gi *= self.ctx.get("intestino", 1.0) * (1.4 if scarso else 1.0)
            disturbo = int(r.random() < p_gi)
            if disturbo:
                self.gi += 1
                premio += ric["disturbo_gi"]
                eventi.append("🤢 disturbo GI")

            # 5. la notte: si dorme peggio in deficit, dopo una gara o con l'intestino in disordine
            p_scarso = 0.25 + (0.2 if self.saldo < -1500 else 0) + (0.15 if tipo == "gara" else 0) \
                + (0.2 if disturbo else 0)
            self.sonno_scarso = int(r.random() < p_scarso)

            self.qualita_tot += qualita
            self.cho_tot += cho
            self.giorno += 1
            fine = self.giorno >= TURNI
            self.ultimo = {"qualità seduta": round(qualita, 2), "gara riuscita": gara_ok,
                           "eventi": ", ".join(eventi)}
            return self.stato(), float(premio), fine

        def esito(self) -> dict:
            n = max(1, self.giorno)
            return {"qualità media seduta": self.qualita_tot / n,
                    "gare riuscite": 100.0 * self.gare_ok / self.gare if self.gare else 0.0,
                    "giornate bassa EA": self.lea, "disturbi GI": self.gi,
                    "giornate in surplus": self.surplus, "CHO medi (g/kg)": self.cho_tot / n}


    def _linee_guida(s):
        """Carboidrati sul tipo di giornata: 3 riposo, 5 leggero, 7 intenso, 9 gara."""
        return (0, 1, 2, 3)[s[0]]


    def _linee_guida_carico(s):
        """Come sopra, più il carico pre-gara (11 g/kg) il giorno prima della gara."""
        return 4 if s[1] else _linee_guida(s)


    POLITICHE_FISSE = {**{f"sempre {a}": (lambda s, i=i: i) for i, a in enumerate(AZIONI[:4])},
                       "📋 linee guida (CHO sul tipo di giornata)": _linee_guida,
                       "📋 linee guida + carico pre-gara": _linee_guida_carico}


    def metriche_episodio(log) -> dict:
        gare = log[log["tipo di giornata"] == "gara"]
        return {"Gare riuscite": f"{int(gare['gara riuscita'].sum())} su {len(gare)}",
                "Giornate critiche": int(log["eventi"].str.contains("bassa").sum()),
                "Disturbi GI": int(log["eventi"].str.contains("GI").sum()),
                "Ricompensa totale": f"{log['ricompensa'].sum():.1f}"}


    def g_episodio(log):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        cho = log["carboidrati"].str.split(" ").str[0].astype(float)
        colori = {"riposo": "#a8c8c4", "leggero": "#5b9c95", "intenso": "#1b5e5a", "gara": "#c2603a"}
        ax.bar(log["giorno"], cho, color=[colori[t] for t in log["tipo di giornata"]], alpha=0.85)
        ax.set_ylabel("carboidrati (g/kg)", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(log["giorno"], log["glicogeno (0–1)"], color="#22303b", linewidth=2, marker="o",
                 markersize=3)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("glicogeno al mattino (0–1)", fontsize=9)
        for _, r in log.iterrows():
            if r["eventi"]:
                ax2.text(r["giorno"], 1.0, "!", ha="center", color="#c2603a", fontsize=10,
                         fontweight="bold")
        from matplotlib.patches import Patch
        ax.legend([Patch(color=c) for c in colori.values()], list(colori), fontsize=7.5,
                  frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
        ax.set_title("Il mese simulato: barre = carboidrati prescritti, linea = glicogeno, ! = evento",
                     fontsize=10, color="#22303b")
        ax.set_xlabel("giorno", fontsize=9)
        for a_ in (ax, ax2):
            a_.spines["top"].set_visible(False)
        ax.set_ylim(0, 12)
        fig.tight_layout()
        return fig

    # γ più basso: l'effetto dei carboidrati di oggi si esaurisce in pochi giorni, e
    # l'agente non vede il calendario (non sa quando finisce il mese)
    IMPOSTAZIONI = {"episodi": 20000, "gamma": 0.80}
    EPISODI_SCELTE = [5000, 10000, 20000, 40000]

    return SimpleNamespace(**locals())


NUTRIZIONE = _ambiente_nutrizione()
