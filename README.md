# Sport AI Lab · Start

Laboratorio didattico di machine learning e reinforcement learning per **chinesiologi e
nutrizionisti**.
È la versione semplificata di *Sport Performance AI Lab*: meno esperimenti e meno opzioni,
lo stesso metodo.

## Avvio

- **Windows** → doppio clic su `avvia.bat`
- **macOS e Linux** → `python3 avvia.py`

`avvia.py` controlla Python, attiva `.venv` se c'è, installa le librerie mancanti, trova una porta
libera dalla 8502 in su e apre il browser. Avvio manuale:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Le cinque sezioni

| sezione | dataset | che cosa si fa |
|---|---|---|
| 🥅 **KPI Sport** | squadra di calcio, una riga per giocatore per settimana (GPS, RPE, sonno, HRV, benessere, prestazione, infortuni) | prevedere RPE, prestazione, rischio di infortunio; trovare profili atletici |
| 🥗 **KPI Nutrizione** | atleti di 5 discipline, una riga per atleta per giorno (diario, colazione, CGM, dispendio, recupero, disponibilità energetica) | prevedere picco glicemico e recupero, allertare la bassa disponibilità energetica; trovare profili alimentari |
| 📂 **I tuoi dati** | un dataset qualsiasi caricato dallo studente (CSV, Excel, JSON) | riconoscere e correggere tipi e ruoli delle colonne, pulire, applicare i cinque modelli adattati alle colonne |
| ⚽ **RL Calcio** | partite simulate, una decisione ogni 5 minuti | un agente impara la tattica in base a punteggio, energia, minuto, pressing avversario, gol subiti, uomini in campo |
| 🍝 **RL Nutrizione** | un mese di preparazione, una decisione al giorno | un agente impara quanti carboidrati prescrivere in base a giornata, gara, glicogeno, bilancio, sonno |

### 📂 I tuoi dati

Lo stesso percorso delle sezioni KPI, su un dataset caricato dallo studente — con colonne
qualsiasi.

1. **Carica** — CSV (separatore e virgola decimale riconosciuti da soli), Excel (si apre il foglio
   con più dati, ma si può sceglierne un altro) o JSON. I numeri scritti come testo («12,5»,
   «1.200», «85%») diventano numeri, sì/no diventa 1/0, le date vengono riconosciute. C'è anche un
   dataset di esempio con errori da pulire.
2. **Analizza e pulisci** — una tabella modificabile con tipo e **ruolo** di ogni colonna (misura,
   categoria, identificativo atleta, ignora), più significato, unità e limiti plausibili scritti
   dallo studente. Pulizia: duplicati, colonne quasi vuote, valori fuori dai limiti, valori estremi
   (Tukey), celle vuote nei numeri e nelle categorie.
3. **Modelli ML** — gli stessi cinque modelli, che si adattano al dataset: il compito si sceglie dal
   bersaglio (regressione, classificazione a due **o più** classi), le categorie diventano colonne
   0/1, divisione per atleta se c'è un identificativo, classi rare escluse, colonne sospette di
   fuga di informazione segnalate. R² e MAE sempre presenti.

### KPI Sport e KPI Nutrizione: tre passi

1. **Genera il dataset** — numero di atleti, durata, variante (seme casuale, una per studente),
   e una quota regolabile di errori realistici: celle vuote, valori impossibili, righe duplicate.
2. **Analizza e pulisci** — tabella del significato di ogni colonna (unità, ruolo, valori
   plausibili), poi pulizia con registro di ciò che è stato fatto.
3. **Modelli ML** — Regressione Lineare, Regressione Logistica, Random Forest, XGBoost, K-Means.
   Si sceglie il modello, poi la variabile bersaglio fra quelle compatibili. Per ogni modello
   l'app spiega come usarlo su quel dataset. Il pulsante *Confronta tutti i modelli* li mette
   uno accanto all'altro sullo stesso bersaglio.

**Ogni analisi riporta R² e MAE**, in verifica e in addestramento (per vedere il
sovradattamento), confrontati con il riferimento banale (la media):

| analisi | R² e MAE calcolati su |
|---|---|
| bersaglio numerico | valore osservato contro valore stimato |
| bersaglio sì/no | evento (0/1) contro probabilità stimata; in più AUC, sensibilità, specificità |
| K-Means | distanza dai centri dei gruppi (R² = varianza spiegata dai gruppi) |
| RL | valore Q (ricompensa prevista dall'agente) contro ricompensa ottenuta davvero |

La verifica è sempre fatta **su atleti mai visti** (divisione per atleta). Le colonne che
contengono già la risposta (es. `carico_interno_ua` per l'RPE, `giorni_stop` per l'infortunio)
vengono escluse in automatico, e l'app spiega perché.

### RL Calcio e RL Nutrizione

Le due sezioni di reinforcement learning hanno lo stesso percorso:

1. **Bonus e malus** — si impostano a mano o partendo da quattro stili già pronti.
2. **Cosa vede l'agente** — si sceglie quali variabili entrano nello stato. Le variabili non
   spuntate continuano a influenzare il gioco, ma l'agente non le vede.
3. **Allena e confronta** — curva di apprendimento, confronto con le strategie fisse su 200
   episodi identici, importanza delle variabili, mappa delle decisioni, R² e MAE dei valori Q,
   storico degli addestramenti per confrontare agenti che vedono variabili diverse.
4. **Episodio e dati** — un episodio passo per passo e tutti i download JSON.

| | RL Calcio | RL Nutrizione |
|---|---|---|
| decisione | tattica ogni 5': pressing alto, blocco medio, difesa bassa, contropiede, possesso | carboidrati ogni giorno: 3, 5, 7, 9, 11 g/kg |
| stato | punteggio, energia, fase, atteggiamento avversario (attendista / equilibrato / pressing alto), gol subito da poco, uomini in campo | tipo di giornata, gara domani, riserve di glicogeno, bilancio energetico, sonno |
| bonus e malus | gol fatto, gol subito, palla recuperata, squadra senza energia, espulsione, vittoria, pareggio, sconfitta | qualità della seduta, gara con riserve piene, bassa disponibilità energetica, disturbo GI, surplus |
| contesto | forza e imprevedibilità dell'avversario | volume di allenamento, sensibilità intestinale |
| confronto con | sempre la stessa tattica, regola dell'allenatore scritta a mano | dosi fisse, linee guida, linee guida + carico pre-gara |

### Download

Ogni tabella si scarica in **JSON**: dataset grezzo e pulito, registro della pulizia, scheda delle
colonne, risultati dei modelli (misure, importanza, previsioni), confronto fra modelli, partita
simulata, storia dell'allenamento, Q-table, tattica appresa, configurazione.

```python
import pandas as pd
df = pd.read_json("sport_dataset_pulito.json")
```

### Didattica

Ogni sezione ha la sua scheda 📚: il percorso, il significato delle colonne, quale modello per
quale scopo, come leggere R² e MAE, la pulizia dei dati, esercizi con consegna e glossario.
Tutti i testi stanno in `didattica.py`.

## Struttura

| file | contenuto |
|---|---|
| `app.py` | tutta l'interfaccia: home e navigazione, KPI Sport e Nutrizione, «I tuoi dati», le due sezioni RL, stile comune |
| `ml.py` | dati simulati (catalogo, generazione, errori, pulizia), lettura e pulizia dei dati dello studente, i cinque modelli con R² e MAE, la versione generica anche multiclasse, grafici |
| `rl.py` | motore di Q-learning e i due ambienti, `rl.CALCIO` e `rl.NUTRIZIONE` |
| `didattica.py` | tutti i testi didattici |
| `avvia.py`, `avvia.bat` | avvio rapido |

Ogni file è diviso in sezioni numerate e segnalate da un'intestazione di commento.
`ml.py` e `rl.py` non importano Streamlit: si usano anche da notebook.

```python
import ml, rl
df = ml.genera("sport", 22, 20, seed=7)
ris = ml.addestra(df, "sport", "Random Forest", "rpe_medio", ml.predittori("sport", df, "rpe_medio"))
print(ris["misure"])

agente = rl.allena(rl.CALCIO, {r.chiave: r.valore for r in rl.CALCIO.RICOMPENSE},
                   {c.chiave: c.valore for c in rl.CALCIO.CONTESTO},
                   rl.Impostazioni(episodi=5000), rl.CALCIO.OSSERVATE_INIZIALI)
```

Le funzioni per i dati dello studente che hanno lo stesso nome di quelle per i dati simulati
portano un suffisso: `pulisci_utente`, `diagnosi_utente`, `addestra_generico`,
`confronta_tutti_generico`, `raggruppa_generico`…

Se `xgboost` non è installato, il modello XGBoost usa il Gradient Boosting di scikit-learn
(stesso principio) e l'app lo segnala.

## Avvertenza

I dati sono **interamente simulati**: ordini di grandezza plausibili, nessuna persona reale. Il
i simulatori delle sezioni RL sono modelli giocattolo. I risultati servono a imparare il metodo e non
sono trasferibili a decisioni su atleti veri.

## Licenza

MIT — © 2026 Christian Fiorino

## File di prova per «I tuoi dati»

Nella cartella `esempi/` ci sono tre file costruiti apposta per mettere alla prova la lettura:
un CSV all'italiana (punto e virgola, virgola decimale, sì/no, date, note libere, un valore
sbagliato), un Excel con un foglio di legenda prima di quello dei dati, e un JSON annidato.
