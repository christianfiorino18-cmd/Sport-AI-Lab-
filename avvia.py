"""
avvia.py — avvio rapido di Sport AI Lab Start.

Un file solo, uguale su Windows, macOS e Linux. Fa in ordine le cinque cose che
altrimenti vanno fatte a mano, e che sono le cinque che tipicamente vanno storte
in aula:

  1. controlla la versione di Python;
  2. attiva l'ambiente virtuale `.venv` se esiste nella cartella;
  3. verifica le librerie e installa quelle mancanti, una volta sola;
  4. cerca una porta libera dalla 8502 in su — così un processo Streamlit
     rimasto appeso da una sessione precedente non fa parlare il browser con
     l'applicazione sbagliata;
  5. apre il browser e avvia Streamlit in modalità headless, che salta la
     richiesta dell'indirizzo email al primo avvio.

Uso:

    python avvia.py                 (su Windows: py avvia.py, oppure avvia.bat)
    python avvia.py --porta 8600    per imporre una porta
    python avvia.py --senza-browser per non aprire la pagina da solo

Copyright (C) 2026  Christian Fiorino — licenza MIT
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

QUI = Path(__file__).resolve().parent
LIBRERIE = ["streamlit", "sklearn", "pandas", "numpy", "matplotlib", "openpyxl"]
PORTA_INIZIALE = 8502


def riga(testo: str = "") -> None:
    print(testo, flush=True)


def intestazione() -> None:
    riga("=" * 46)
    riga("  Sport AI Lab Start")
    riga("=" * 46)
    riga()


# ------------------------------------------------------------------ ambiente
def python_del_venv() -> str | None:
    """Se c'è un `.venv` nella cartella, uso il suo interprete."""
    candidati = [QUI / ".venv" / "Scripts" / "python.exe",      # Windows
                 QUI / ".venv" / "bin" / "python"]              # macOS e Linux
    for c in candidati:
        if c.exists():
            return str(c)
    return None


def librerie_mancanti(python: str) -> list[str]:
    codice = ("import importlib.util as u, sys;"
              "print(','.join(m for m in sys.argv[1:] if u.find_spec(m) is None))")
    esito = subprocess.run([python, "-c", codice, *LIBRERIE],
                           capture_output=True, text=True)
    return [m for m in esito.stdout.strip().split(",") if m]


def installa(python: str) -> bool:
    riga("Mancano alcune librerie. Le installo adesso (una sola volta)…")
    riga()
    esito = subprocess.run([python, "-m", "pip", "install", "-r",
                            str(QUI / "requirements.txt")])
    return esito.returncode == 0


# --------------------------------------------------------------------- porta
def libera(porta: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.35)
        return s.connect_ex(("127.0.0.1", porta)) != 0


def prima_porta_libera(inizio: int = PORTA_INIZIALE) -> int:
    for porta in range(inizio, inizio + 40):
        if libera(porta):
            return porta
    return inizio


# ---------------------------------------------------------------------- main
def main(argomenti: list[str]) -> int:
    intestazione()

    if sys.version_info < (3, 9):
        riga(f"[ERRORE] Serve Python 3.9 o superiore (in uso: "
             f"{sys.version.split()[0]}).")
        return 1

    python = python_del_venv()
    if python:
        riga(f"Ambiente virtuale trovato: {Path(python).parent.parent.name}")
    else:
        python = sys.executable
    riga(f"Python in uso: {python}")

    mancanti = librerie_mancanti(python)
    if mancanti:
        riga()
        riga("Mancano: " + ", ".join(mancanti))
        if not installa(python):
            riga()
            riga("[ERRORE] Installazione non riuscita. Le righe qui sopra dicono "
                 "perché.")
            return 1
        riga()

    try:
        prova = subprocess.run([python, "-c", "import xgboost"], capture_output=True)
        if prova.returncode != 0:
            riga("[NOTA] xgboost non installato: il modello XGBoost userà il Gradient "
                 "Boosting di scikit-learn.")
            riga("       Per la versione originale: pip install xgboost")
            riga()
    except Exception:
        pass

    porta = (int(argomenti[argomenti.index("--porta") + 1])
             if "--porta" in argomenti else prima_porta_libera())
    indirizzo = f"http://localhost:{porta}"
    riga(f"Porta scelta: {porta}")
    riga(f"Apro il browser su {indirizzo}")
    riga("Per fermare l'applicazione: chiudi questa finestra oppure premi Ctrl+C.")
    riga()

    if "--senza-browser" not in argomenti:
        threading.Timer(2.5, lambda: webbrowser.open(indirizzo)).start()

    ambiente = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1")
    comando = [python, "-m", "streamlit", "run", str(QUI / "app.py"),
               "--server.port", str(porta),
               "--server.address", "localhost",
               "--server.headless", "true",
               "--browser.gatherUsageStats", "false"]
    try:
        esito = subprocess.run(comando, cwd=str(QUI), env=ambiente)
        codice = esito.returncode
    except KeyboardInterrupt:
        codice = 0

    riga()
    riga("=" * 46)
    riga("  Streamlit si è fermato.")
    riga("  Se sopra compare un messaggio di errore, è quello il motivo.")
    riga("=" * 46)
    return codice


if __name__ == "__main__":
    codice = main(sys.argv[1:])
    if os.name == "nt" and "--no-pausa" not in sys.argv:
        try:
            input("\nPremi Invio per chiudere. ")
        except EOFError:
            pass
    raise SystemExit(codice)
