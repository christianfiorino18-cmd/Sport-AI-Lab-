"""
didattica.py — Sport AI Lab Start · testi della scheda Didattica

Tutto il materiale didattico in un posto solo, separato dall'interfaccia:
per aggiornare una spiegazione si modifica soltanto questo file.

Copyright (C) 2026  Christian Fiorino — licenza MIT
"""
from __future__ import annotations

# =========================================================================
# I MODELLI: cosa fanno e come si usano NEI NOSTRI DATASET
# =========================================================================

COSA_FA = {
    "Regressione Lineare": (
        "Traccia la retta (o il piano) che passa più vicino a tutti i punti: "
        "bersaglio = a + b₁·x₁ + b₂·x₂ + … Ogni coefficiente dice di quanto cambia il "
        "bersaglio quando quella variabile aumenta, a parità delle altre."),
    "Regressione Logistica": (
        "Come la lineare, ma il risultato passa per una curva a S che lo trasforma in una "
        "probabilità fra 0 e 1. Serve per bersagli sì/no. I coefficienti dicono se una "
        "variabile alza o abbassa la probabilità dell'evento."),
    "Random Forest": (
        "Centinaia di alberi decisionali («se ACWR > 1,4 e sonno < 6,5 h allora…»), ciascuno "
        "addestrato su un campione diverso dei dati. La previsione finale è la media (o il voto "
        "a maggioranza) degli alberi. Coglie soglie e interazioni senza che le si dichiari."),
    "XGBoost": (
        "Anche questo usa alberi, ma in sequenza: ogni nuovo albero corregge gli errori dei "
        "precedenti (gradient boosting). Spesso è il più preciso sui dati tabellari, ma è anche "
        "quello che più facilmente impara a memoria i dati di addestramento."),
    "K-Means": (
        "Non ha un bersaglio: cerca gruppi di atleti simili. Sceglie k centri, assegna ogni "
        "atleta al centro più vicino, sposta i centri nella media del gruppo e ripete finché "
        "nulla cambia. I gruppi vanno poi interpretati e battezzati da chi conosce lo sport."),
}

USO = {
    "sport": {
        "Regressione Lineare": (
            "**Bersagli**: `rpe_medio`, `indice_prestazione`.  \n"
            "Prevedere l'**RPE** dal carico esterno GPS (distanza, alta intensità, sprint) e dal "
            "benessere è il caso ideale: il legame è quasi lineare, e i coefficienti si leggono "
            "come «ogni km in più vale +0,1 punti di RPE». Se il giocatore dichiara un RPE molto "
            "più alto di quello previsto, è un segnale di fatica da indagare."),
        "Regressione Logistica": (
            "**Bersaglio**: `infortunio`.  \n"
            "Stima la probabilità di infortunio nelle due settimane successive da ACWR, sonno, HRV, "
            "età. È il punto di partenza obbligato: semplice, leggibile, e i coefficienti mostrano "
            "subito se il rischio sale con l'ACWR come dice la letteratura."),
        "Random Forest": (
            "**Bersagli**: tutti (`rpe_medio`, `indice_prestazione`, `infortunio`).  \n"
            "Utile quando il rischio non cresce in modo lineare ma oltre una soglia (ACWR sopra "
            "1,3–1,5). Il grafico dell'importanza dice quali variabili lo staff dovrebbe "
            "monitorare per prime."),
        "XGBoost": (
            "**Bersagli**: tutti.  \n"
            "Il candidato per la massima precisione su `indice_prestazione` e `rpe_medio`. "
            "Sul bersaglio `infortunio` (pochi eventi) tende a imparare a memoria: guarda la "
            "distanza fra R² di addestramento e R² di verifica."),
        "K-Means": (
            "**Nessun bersaglio**: si raggruppano i giocatori sulle loro medie stagionali.  \n"
            "Esempio: distanza, alta intensità, sprint, velocità massima → emergono profili "
            "atletici (gli «sprinter», i «motori», i giocatori a basso volume) che spesso, ma non "
            "sempre, coincidono con il ruolo. I giocatori «fuori ruolo» sono i più interessanti."),
    },
    "nutrizione": {
        "Regressione Lineare": (
            "**Bersagli**: `picco_glicemico`, `recupero`.  \n"
            "Il picco glicemico dopo colazione dipende da carboidrati, indice glicemico, fibra e "
            "sonno: la lineare dice quanti mg/dl vale ogni 10 g di carboidrati in più e quanto "
            "li compensa la fibra. È il modello più semplice da spiegare a un atleta."),
        "Regressione Logistica": (
            "**Bersaglio**: `bassa_ea`.  \n"
            "Stima la probabilità che una giornata finisca sotto i 30 kcal/kg FFM di disponibilità "
            "energetica, usando solo ciò che si sa al mattino (colazione, tipo di giornata, "
            "variazione di peso sulla bilancia, "
            "dispendio, profilo). Le colonne del diario completo sono escluse: ricalcolerebbero "
            "la formula invece di prevedere."),
        "Random Forest": (
            "**Bersagli**: tutti (`picco_glicemico`, `recupero`, `bassa_ea`).  \n"
            "Coglie le interazioni che la lineare non vede: per esempio che la stessa colazione "
            "alza di più la glicemia dopo una notte corta. L'importanza indica le leve "
            "nutrizionali più forti."),
        "XGBoost": (
            "**Bersagli**: tutti.  \n"
            "Spesso il più preciso sul `recupero`, che dipende da molte cose insieme (sonno, "
            "carboidrati rispetto al fabbisogno, proteine, idratazione). Da confrontare sempre con "
            "la lineare: se la differenza è piccola, vince il modello più semplice."),
        "K-Means": (
            "**Nessun bersaglio**: si raggruppano gli atleti sulle loro abitudini medie.  \n"
            "Esempio: carboidrati g/kg, proteine g/kg, disponibilità energetica, picco glicemico → "
            "emergono profili come «restrittivi», «alti carboidrati», «alta risposta glicemica», "
            "utili per personalizzare l'educazione alimentare."),
    },
}

QUALE_SCOPO = {
    "sport": [
        ("Spiegare allo staff da cosa dipende l'RPE", "Regressione Lineare",
         "coefficienti leggibili, R² già alto"),
        ("Stimare il rischio di infortunio di un giocatore", "Regressione Logistica",
         "probabilità diretta e leggibile; poi confronta con Random Forest"),
        ("Trovare le soglie di rischio (es. ACWR > 1,4)", "Random Forest",
         "gli alberi lavorano proprio per soglie"),
        ("Ottenere la previsione più precisa della prestazione", "XGBoost",
         "di solito il più accurato, se non sovradatta"),
        ("Creare profili atletici per allenamenti differenziati", "K-Means",
         "non serve un bersaglio, solo misure simili"),
    ],
    "nutrizione": [
        ("Quanto alza la glicemia ogni grammo di carboidrati", "Regressione Lineare",
         "un coefficiente per variabile, in mg/dl"),
        ("Allertare il mattino per una giornata a bassa disponibilità energetica",
         "Regressione Logistica", "probabilità leggibile, soglia di allarme regolabile"),
        ("Scoprire le interazioni fra sonno, pasto e glicemia", "Random Forest",
         "coglie effetti combinati senza dichiararli"),
        ("Prevedere il recupero con la massima precisione", "XGBoost",
         "molte variabili che agiscono insieme"),
        ("Dividere gli atleti per abitudini alimentari", "K-Means",
         "gruppi da interpretare con l'occhio del nutrizionista"),
    ],
}


# =========================================================================
# R² E MAE
# =========================================================================

METRICHE = """
**MAE — errore medio assoluto.** Di quanto sbaglia il modello, in media, *nelle unità del
bersaglio*. MAE = 0,4 su `rpe_medio` vuol dire «sbaglio in media di 0,4 punti di RPE»;
MAE = 9 su `picco_glicemico` vuol dire «sbaglio di 9 mg/dl». Più basso è meglio. È la misura
da comunicare a un allenatore o a un atleta, perché si capisce senza statistica.

**R² — varianza spiegata.** Quanta parte delle differenze fra una riga e l'altra il modello
riesce a spiegare. 1 = perfetto, 0 = come usare sempre la media, **negativo = peggio della
media**. È una misura relativa: permette di confrontare bersagli con unità diverse.

| R² in verifica | lettura indicativa |
|---|---|
| sopra 0,7 | il modello coglie bene il fenomeno |
| 0,3 – 0,7 | utile, ma molta variabilità resta inspiegata |
| 0 – 0,3 | debole: il fenomeno dipende da cose che non misuriamo |
| sotto 0 | inutile: meglio la media |

**Addestramento contro verifica.** L'app riporta entrambe. Se R² in addestramento è 0,95 e in
verifica 0,30, il modello ha **imparato a memoria** (sovradattamento, *overfitting*). Conta
solo il numero in verifica.

**La verifica è fatta su atleti mai visti.** I dati vengono divisi *per atleta*: chi sta
nell'insieme di verifica non compare mai in addestramento. È la domanda onesta: il modello
funziona su un giocatore nuovo? Una divisione casuale mescolerebbe settimane dello stesso
giocatore nelle due parti e darebbe numeri più belli ma falsi.

**R² e MAE per un bersaglio sì/no** (infortunio, bassa_ea). Il modello dà una probabilità
(0,12 = 12%); R² e MAE sono calcolati fra quella probabilità e ciò che è successo (0 o 1).
Un R² vicino a 0 qui è normale: prevedere eventi rari è difficile. Per questo l'app aggiunge
**AUC** (0,5 = moneta, 1 = perfetto: quanto bene il modello mette in cima i casi veri),
**sensibilità** (quanti eventi veri l'allarme ha preso) e **specificità** (quanti non-eventi
ha lasciato tranquilli). L'accuratezza da sola inganna: con il 10% di infortuni, dire sempre
«nessun infortunio» ha il 90% di accuratezza e non serve a niente.

**R² e MAE per il K-Means.** Non c'è un bersaglio, quindi le misure riguardano la compattezza
dei gruppi: **R²** = quota delle differenze fra atleti spiegata dall'appartenenza al gruppo;
**MAE** = distanza media di ogni valore dal centro del suo gruppo, in deviazioni standard.
Aumentando k l'R² sale sempre: si sceglie il k oltre il quale guadagna poco (il «gomito»).

**R² e MAE per il reinforcement learning.** Il valore Q è la previsione dell'agente sulla
ricompensa futura. Confrontandolo con quella ottenuta davvero si misura quanto l'agente
«conosce» la partita. Nel calcio l'R² resta basso: un gol è un evento raro e casuale.
"""

PULIZIA = """
I dati reali hanno sempre tre difetti. L'app li inserisce di proposito nel passo 1 (si possono
disattivare) per allenarsi a trovarli.

| difetto | esempio | che cosa fare |
|---|---|---|
| **righe duplicate** | lo stesso file GPS importato due volte | eliminarle: contano doppio un giocatore |
| **valori impossibili** | 280 km in una settimana, sonno negativo | sono errori, non record: trasformarli in celle vuote o eliminare la riga |
| **celle vuote** | il giocatore non ha compilato il questionario | sostituire con la mediana *dello stesso atleta*, oppure eliminare la riga |

**Perché la mediana e non la media?** La mediana non è trascinata dai valori estremi.
**Perché quella dello stesso atleta?** Perché il sonno di un giocatore somiglia al suo sonno
delle altre settimane più che a quello della squadra.

**Eliminare o sostituire?** Eliminare è più onesto ma butta via dati; sostituire li conserva
ma inventa un valore plausibile. Con poche celle vuote le due scelte danno risultati simili:
provale entrambe e confronta R² e MAE.

**I limiti plausibili** di ogni colonna sono nella tabella del significato delle colonne
(colonna «valori plausibili»). Sono una decisione di dominio, non statistica.

**Colonne escluse in automatico.** Alcune colonne contengono già la risposta (*fuga di
informazione*). Esempi: `carico_interno_ua` è RPE × minuti, quindi non può servire a prevedere
l'RPE; `giorni_stop` è noto solo dopo l'infortunio. Regola pratica: **avrei questo numero in
mano *prima* dell'evento?** Se no, va tolto.
"""

PERCORSO = """
1. **Genera il dataset.** Scegli quanti atleti e per quanto tempo. La *variante* cambia i numeri
   casuali: dai una variante diversa a ogni studente, così nessuno copia.
2. **Analizza e pulisci.** Leggi il significato di ogni colonna, poi trova e correggi duplicati,
   valori impossibili e celle vuote.
3. **Usa i modelli.** Scegli un modello e un bersaglio, guarda R² e MAE, poi premi
   *Confronta tutti i modelli* per vedere quale funziona meglio sullo stesso problema.

Ogni tabella si scarica in formato **JSON**, leggibile con Python (`pandas.read_json`).
"""

ESERCIZI = {
    "sport": [
        ("RPE dal GPS", "Prevedi `rpe_medio` con la Regressione Lineare. Quale variabile ha il "
         "coefficiente più grande? Ripeti includendo solo le variabili GPS.",
         "R² e MAE delle due prove e una frase di commento."),
        ("Lineare contro alberi", "Su `indice_prestazione` premi «Confronta tutti i modelli». "
         "Quale ha il maggiore distacco fra R² di addestramento e di verifica?",
         "La tabella del confronto in JSON e due righe sul sovradattamento."),
        ("La soglia di allarme", "Prevedi `infortunio` con la Logistica. Porta la soglia da 0,5 a "
         "0,2: come cambiano sensibilità e falsi allarmi?",
         "Una tabella con le due soglie e la scelta motivata per uno staff medico."),
        ("Pulire o eliminare?", "Rigenera il dataset con 10% di celle vuote. Prova le due "
         "strategie di pulizia e confronta l'R² su `rpe_medio`.",
         "Il registro della pulizia e le due misure."),
        ("Profili atletici", "Usa K-Means con distanza, alta intensità, sprint e velocità massima. "
         "Prova k da 2 a 5 e scegli con il gomito.",
         "La tabella dell'identikit e un nome per ogni gruppo."),
    ],
    "nutrizione": [
        ("Il peso della colazione", "Prevedi `picco_glicemico` con la Regressione Lineare. "
         "Quanti mg/dl valgono 10 g di carboidrati in più? E la fibra?",
         "R², MAE e i due coefficienti commentati."),
        ("Il recupero", "Confronta tutti i modelli su `recupero`. Il più preciso vale la "
         "complessità in più rispetto alla lineare?",
         "La tabella del confronto in JSON e un giudizio motivato."),
        ("Allarme mattutino", "Prevedi `bassa_ea` con la Logistica e con Random Forest. "
         "Quale ha l'AUC più alta? Quale variabile pesa di più?",
         "Le due matrici di confusione e la scelta del modello."),
        ("La fuga di informazione", "Leggi l'elenco delle colonne escluse per `bassa_ea`. "
         "Spiega con parole tue perché ciascuna tradirebbe la risposta.",
         "Mezza pagina."),
        ("Profili alimentari", "Usa K-Means su carboidrati g/kg, proteine g/kg, disponibilità "
         "energetica e picco glicemico. Dai un nome ai gruppi.",
         "L'identikit dei gruppi e la composizione per sport."),
    ],
}

GLOSSARIO = [
    ("Dataset", "Una tabella: ogni riga è un'osservazione (un giocatore in una settimana), ogni "
     "colonna una variabile."),
    ("Bersaglio (target)", "La colonna che vogliamo prevedere: RPE, picco glicemico, infortunio…"),
    ("Variabili in ingresso (predittori)", "Le colonne che il modello usa per prevedere il bersaglio."),
    ("Regressione", "Prevedere un numero (RPE 6,4; picco 132 mg/dl)."),
    ("Classificazione", "Prevedere una categoria, qui sì/no (infortunio, bassa disponibilità energetica)."),
    ("Raggruppamento (clustering)", "Cercare gruppi simili senza un bersaglio. Esempio: K-Means."),
    ("Addestramento / verifica", "Il modello impara su una parte dei dati e viene giudicato "
     "sull'altra, che non ha mai visto."),
    ("Sovradattamento (overfitting)", "Il modello impara a memoria l'addestramento e sbaglia sui "
     "dati nuovi: R² di addestramento molto più alto di quello di verifica."),
    ("Fuga di informazione (leakage)", "Una colonna in ingresso contiene già la risposta. Il "
     "punteggio diventa ottimo e il modello inutile."),
    ("Standardizzazione", "Portare ogni variabile a media 0 e deviazione standard 1, per "
     "confrontare km e ore di sonno sulla stessa scala."),
    ("One-hot", "Trasformare una categoria (ruolo) in tante colonne 0/1, una per valore."),
    ("Importanza delle variabili", "Quanto ogni colonna contribuisce alle previsioni del modello."),
    ("Soglia di allarme", "La probabilità oltre la quale il modello dice «sì». Più bassa = più "
     "eventi presi ma più falsi allarmi."),
    ("ACWR", "Acute:Chronic Workload Ratio: carico della settimana diviso per la media delle "
     "precedenti. Sopra ~1,5 il rischio di infortunio sale."),
    ("Disponibilità energetica", "(energia introdotta − energia spesa in allenamento) / massa "
     "magra. Sotto 30 kcal/kg FFM è bassa."),
    ("Reinforcement learning", "Un agente impara per tentativi: prova un'azione, riceve un bonus "
     "o un malus, aggiorna la sua strategia."),
    ("Stato · azione · ricompensa", "Ciò che l'agente osserva · ciò che sceglie · il bonus/malus "
     "che ne riceve."),
    ("Q-table", "La tabella in cui l'agente annota quanto vale ogni azione in ogni stato."),
    ("Esplorazione (ε)", "La quota di scelte fatte a caso per provare cose nuove invece di "
     "ripetere quella che sembra migliore."),
    ("Sconto (γ)", "Quanto conta il futuro: vicino a 1, l'agente pensa anche alla fine della partita."),
]


# =========================================================================
# REINFORCEMENT LEARNING
# =========================================================================

GLOSSARIO_RL = [
    ("Reinforcement learning", "Un agente impara per tentativi: prova un'azione, riceve un bonus "
     "o un malus, aggiorna la sua strategia. Nessuno gli dice qual è la scelta giusta."),
    ("Agente", "Chi decide: l'allenatore virtuale o il nutrizionista virtuale."),
    ("Stato", "Ciò che l'agente vede prima di decidere: l'insieme delle variabili osservate."),
    ("Azione", "La scelta: una tattica, una quantità di carboidrati."),
    ("Ricompensa", "La somma di bonus e malus che segue un'azione."),
    ("Episodio", "Una partita intera o un mese intero di preparazione."),
    ("Politica", "La regola appresa: per ogni stato, quale azione scegliere."),
    ("Q-table", "La tabella in cui l'agente annota quanto vale ogni azione in ogni stato."),
    ("Esplorazione (ε)", "La quota di scelte fatte a caso per provare cose nuove invece di "
     "ripetere quella che sembra migliore."),
    ("Sconto (γ)", "Quanto conta il futuro: vicino a 1 l'agente pensa a lungo termine, vicino a 0 "
     "solo alla prossima decisione."),
    ("Variabile nascosta", "Una variabile che influenza il gioco ma che l'agente non vede: lui "
     "la subisce come se fosse caso."),
    ("Maledizione della dimensionalità", "Ogni variabile in più moltiplica le situazioni da "
     "imparare: servono molti più episodi per vederle tutte abbastanza volte."),
]

RL_COMUNE = """
#### Come impara: Q-learning

L'agente tiene una tabella Q con un valore per ogni coppia (situazione, azione). All'inizio è
tutta a zero. Dopo ogni decisione aggiorna la casella usata:

> Q(s, a) ← Q(s, a) + α · [ ricompensa + γ · max Q(s', ·) − Q(s, a) ]

- **α**: quanto si fida dell'ultima esperienza. Parte da 1 e scende con il numero di volte che ha
  già visto quella situazione, fino al minimo che imposti.
- **γ**: quanto conta il futuro.
- **ε**: quota di scelte a caso per esplorare; cala con il passare degli episodi.

#### Come leggere i risultati

| cosa | domanda a cui risponde |
|---|---|
| curva di apprendimento | l'agente migliora con l'esperienza? |
| confronto con le strategie fisse | adattare la scelta alla situazione rende davvero? |
| importanza delle variabili | quali informazioni cambiano davvero la decisione? |
| mappa delle decisioni | che cosa sceglie in ogni combinazione di due variabili? |
| R² e MAE dei valori Q | quanto bene l'agente prevede la ricompensa che otterrà? |
| storico degli addestramenti | che cosa cambia se vede più o meno variabili, o con altri bonus/malus? |

Le caselle «—» della mappa sono situazioni incontrate meno di 25 volte: lì l'agente non ha
imparato abbastanza e la sua scelta non vale.

#### Vedere di più non è gratis

Ogni variabile in più moltiplica le situazioni da imparare. Con poche variabili l'agente impara in
fretta ma decide «alla cieca» su ciò che non vede; con tutte le variabili può decidere su misura,
ma ha bisogno di molti più episodi. Su un atleta vero quegli episodi non esistono: è il motivo per
cui il RL in sport si allena su simulatori, e il simulatore va validato.

**Avvertenza.** I due ambienti sono modelli giocattolo: le probabilità sono plausibili ma
inventate. Servono a capire come ragiona un agente, non a scegliere la tattica di domenica o la
dieta di un atleta.
"""

RL = {
    "calcio": {
        "intro": "Un allenatore virtuale sceglie la tattica ogni 5 minuti — pressing alto, blocco "
                 "medio, difesa bassa, contropiede, possesso palla — guardando punteggio, energia, "
                 "minuto, atteggiamento dell'avversario, gol appena subiti e uomini in campo. "
                 "<b>Bonus e malus li decidi tu; le variabili che vede, anche.</b>",
        "intro_lunga": """
Un allenatore virtuale (l'**agente**) sceglie la tattica ogni 5 minuti di una partita da 90'.
Non sa nulla di calcio: prova, osserva cosa succede, riceve bonus e malus e, partita dopo
partita, capisce quale tattica conviene in ogni situazione.

La partita cambia mentre la si gioca: l'avversario **alza il pressing** quando perde e **si chiude**
quando vince; dopo un **gol subito** la squadra è scossa per 10 minuti; un'**espulsione** cambia
tutto. Se l'agente vede queste variabili può reagire; se non le vede, le subisce come se fossero
sfortuna.
""",
        "stato": "Ogni variabile spuntata entra nello **stato**: l'agente impara una decisione "
                 "diversa per ogni combinazione. Togli la spunta a *atteggiamento avversario* e "
                 "l'agente non saprà più se l'avversario sta pressando: dovrà scegliere una tattica "
                 "che vada bene «in media».",
        "azioni": """
#### Le tattiche

| tattica | gol fatti | gol subiti | energia | quando rende |
|---|---|---|---|---|
| Pressing alto | alti | alti, altissimi se stanchi o in 10 | molta | squadra fresca, avversario chiuso |
| Blocco medio | medi | medi | media | soluzione di equilibrio |
| Difesa bassa | pochi | pochi | poca | in inferiorità numerica, a difendere un vantaggio |
| Contropiede | medi | medi | media | **contro chi alza il pressing** o si sbilancia perché perde |
| Possesso palla | medi | pochi | poca | per riprendere il controllo dopo un gol subito; **rischioso contro il pressing** |
""",
        "esercizi": [
            ("Che cosa vede l'agente", "Allena l'agente con le sole variabili punteggio, energia e "
             "fase. Poi aggiungi atteggiamento avversario e riallena con gli stessi episodi.",
             "Lo storico degli addestramenti in JSON e due righe: quanto vale sapere cosa fa "
             "l'avversario?"),
            ("Contro il pressing", "Con tutte le variabili, guarda «Come cambia la decisione al "
             "variare di… atteggiamento avversario». Che cosa sceglie l'agente quando l'avversario "
             "pressa?", "La tabella e un confronto con ciò che faresti tu."),
            ("Dopo il gol subito", "Nella mappa metti in righe «gol subito da poco» e in colonne "
             "«fase». Come reagisce l'agente?", "La mappa e un commento."),
            ("Il catenaccio", "Usa lo stile «Catenaccio» e riallena. Che cosa succede alla difesa "
             "bassa in 10 contro 11?", "Le due mappe (equilibrato e catenaccio)."),
            ("Più variabili, più episodi", "Con tutte le variabili, allena con 5.000 e poi con "
             "80.000 episodi. Chi batte la regola dell'allenatore?",
             "Lo storico e mezza pagina sulla maledizione della dimensionalità."),
        ],
    },
    "nutri": {
        "intro": "Un nutrizionista virtuale sceglie ogni giorno i carboidrati (3–11 g/kg) per un "
                 "mese di preparazione, guardando tipo di giornata, gara del giorno dopo, riserve "
                 "di glicogeno, bilancio energetico e sonno. <b>Bonus e malus li decidi tu; le "
                 "variabili che vede, anche.</b>",
        "intro_lunga": """
Un nutrizionista virtuale (l'**agente**) segue un atleta di endurance di 70 kg per quattro
settimane. Ogni mattina decide quanti carboidrati prescrivere. Nessuno gli dice le linee guida:
vede solo le conseguenze — sedute buone o cattive, gare con le riserve piene o vuote, giornate in
bassa disponibilità energetica, disturbi gastrointestinali, aumento di peso.

Il confronto con le **linee guida** (carboidrati in base al tipo di giornata) e con le **linee
guida + carico pre-gara** dice se l'agente le ha ritrovate da solo, e se ha imparato qualcosa in
più: per esempio a dare meno carboidrati dopo una notte insonne, o a recuperare quando le riserve
sono basse.

**Perché l'agente non carica il giorno della gara?** Nel simulatore, come nella realtà, la
gara si corre con il glicogeno accumulato *prima*: i carboidrati del giorno di gara servono a
recuperare, non a correre. Un buon agente impara a caricare il giorno prima (se vede la variabile
«domani»), non il giorno stesso.

**Perché γ = 0,8?** L'effetto dei carboidrati di oggi si esaurisce in pochi giorni, e l'agente non
vede il calendario: con γ vicino a 1 crederebbe che il mese non finisca mai e le sue previsioni
(i valori Q) sarebbero molto meno precise. Prova a cambiarlo e guarda R² e MAE dei valori Q.
""",
        "stato": "Ogni variabile spuntata entra nello **stato**: l'agente impara una prescrizione "
                 "diversa per ogni combinazione. Togli la spunta a *domani* e non saprà più che "
                 "domani c'è gara: non potrà imparare il carico pre-gara.",
        "azioni": """
#### Le prescrizioni e le loro conseguenze

| carboidrati | conseguenze tipiche |
|---|---|
| 3 g/kg | adeguati al riposo; sotto il fabbisogno nelle giornate dure → riserve vuote, deficit |
| 5 g/kg | giornate leggere |
| 7 g/kg | giornate intense |
| 9 g/kg | gara; il giorno prima della gara contribuisce al carico |
| 11 g/kg | carico pre-gara; se non serve → disturbi gastrointestinali e surplus |

Il **sonno scarso** rallenta la risintesi del glicogeno e rende l'intestino più sensibile. Il
**bilancio energetico** accumula gli ultimi giorni: in deficit si va in bassa disponibilità
energetica, in surplus si prende peso.
""",
        "esercizi": [
            ("Le linee guida ritrovate", "Allena l'agente con la sola variabile «tipo di "
             "giornata». Che carboidrati sceglie per riposo, leggero, intenso e gara?",
             "La tabella delle decisioni e il confronto con le linee guida."),
            ("Il carico pre-gara", "Aggiungi la variabile «domani» e riallena. L'agente aumenta i "
             "carboidrati il giorno prima della gara?", "Lo storico e la tabella «Come cambia la "
             "decisione al variare di… domani»."),
            ("Salute prima di tutto", "Usa lo stile «Salute prima di tutto». Come cambiano le "
             "giornate in bassa disponibilità energetica e i disturbi GI?",
             "Il confronto in JSON e un commento."),
            ("Intestino delicato", "Porta la sensibilità intestinale a 1,5. L'agente rinuncia al "
             "carico pre-gara?", "Due mappe a confronto."),
            ("Chi decide la penalità?", "Il malus della bassa disponibilità energetica è un "
             "parametro tecnico o una scelta clinica? Chi dovrebbe deciderlo?", "Mezza pagina."),
        ],
    },
}


# =========================================================================
# I TUOI DATI
# =========================================================================

USO_GENERICO = {
    "Regressione Lineare": (
        "**Bersaglio**: una colonna numerica con molti valori diversi (salto in cm, tempo di "
        "sprint, picco glicemico…).  \n"
        "Il primo modello da provare: dice di quanto cambia il bersaglio per ogni unità di ciascuna "
        "variabile. **Attenzione ai valori estremi**: un solo errore di battitura (12000 invece di "
        "1200) può rovinarlo, perché la retta si piega per raggiungerlo."),
    "Regressione Logistica": (
        "**Bersaglio**: una colonna sì/no o con poche categorie (infortunio sì/no, livello "
        "amatoriale/regionale/elite…).  \n"
        "Stima la probabilità di ogni classe. Con più di due classi diventa *multinomiale*: una "
        "probabilità per classe, che sommate fanno 1."),
    "Random Forest": (
        "**Bersaglio**: numerico o a categorie.  \n"
        "Si adatta bene a quasi ogni dataset: non soffre i valori estremi, coglie soglie e "
        "interazioni. Il grafico dell'importanza dice quali colonne contano davvero."),
    "XGBoost": (
        "**Bersaglio**: numerico o a categorie.  \n"
        "Spesso il più preciso, ma con pochi dati (meno di qualche centinaio di righe) tende a "
        "imparare a memoria: guarda sempre la distanza fra R² di addestramento e di verifica."),
    "K-Means": (
        "**Nessun bersaglio**: scegli due o più colonne numeriche e il numero di gruppi.  \n"
        "Se il file ha più righe per la stessa persona, conviene raggruppare **una riga per "
        "atleta** (la media delle sue misure): altrimenti si raggruppano le sedute, non le persone."),
}

PREPARA_FILE = """
#### Come preparare il file

| regola | perché |
|---|---|
| **una riga = un'osservazione** (un atleta in un test, una seduta, una giornata) | è ciò che il modello impara a prevedere |
| **una sola riga di intestazione**, con i nomi delle colonne | le celle unite e i titoli su più righe confondono la lettura (puoi indicare in quale riga sta l'intestazione) |
| **una colonna con il codice dell'atleta** se ci sono più righe per persona | permette la verifica onesta *su atleti mai visti* |
| **numeri come numeri** | «12,5», «1.200», «85%» vengono convertiti da soli; «circa 12» o «12 cm» no: diventano testo |
| **niente righe di totali o medie** in fondo al foglio | verrebbero lette come atleti |
| **niente dati identificativi** (nomi, codici fiscali, date di nascita complete) | usa codici anonimi: sono dati sanitari, tutelati dal GDPR |

Formati accettati: **CSV** (separatore e decimali riconosciuti da soli: punto e virgola e virgola
decimale all'italiana vanno bene), **Excel** (.xlsx, scegli il foglio) e **JSON**.

L'app gira sul tuo computer: i file non vengono inviati da nessuna parte.
"""

RUOLI_SPIEGATI = """
#### I ruoli delle colonne

| ruolo | che cosa vuol dire | esempio |
|---|---|---|
| **misura** | numero usabile come ingresso o come bersaglio | CMJ (cm), RPE, kcal |
| **categoria** | etichetta con pochi valori: diventa una colonna 0/1 per ogni valore | ruolo, sesso, livello, infortunio sì/no |
| **identificativo atleta** | dice a chi appartiene la riga; non entra nel modello, serve a dividere i dati per persona | «Codice atleta», «ID» |
| **ignora** | non entra nel modello | date, note libere, colonne calcolate dal bersaglio |

L'app propone un ruolo per ogni colonna; **correggilo tu**: è una decisione di dominio. Un numero
che è in realtà un codice (ruolo in campo 1, 2, 3, 4) va messo come *categoria*, altrimenti il
modello crederebbe che «4» sia il doppio di «2».

Puoi anche scrivere il **significato** e l'**unità** di ogni colonna e i **limiti plausibili**:
i valori fuori dai limiti vengono trattati come errori nella pulizia. È la scheda delle variabili
del tuo dataset, e la puoi scaricare.
"""

ADATTAMENTO = """
#### Come si adattano i modelli ai tuoi dati

- **Il compito si sceglie dal bersaglio**: un numero con molti valori diversi → regressione; sì/no
  o poche categorie → classificazione (anche con più di due classi). Per un numero con pochi
  valori (es. un punteggio da 1 a 5) puoi scegliere tu come trattarlo.
- **Colonne di testo e categorie** → una colonna 0/1 per ogni valore (*one-hot*). **Colonne
  numeriche** → standardizzate. **Celle vuote rimaste** → mediana o categoria più frequente.
- **Divisione per atleta** se c'è una colonna identificativa con almeno 4 persone, altrimenti
  **casuale** (mantenendo le proporzioni delle classi).
- **Classi con meno di 5 casi** vengono tolte: non si può imparare da 2 esempi.
- **Meno di 20 righe utilizzabili** → l'analisi si ferma: non è un limite del programma, è un
  limite della statistica.
- **Possibile fuga di informazione**: le colonne quasi identiche al bersaglio (correlazione oltre
  0,95, o categorie che lo determinano senza eccezioni) vengono segnalate.

#### R² e MAE quando le classi sono più di due

Il modello dà una probabilità per ogni classe. **MAE** = in media, quanta probabilità il modello
lascia *fuori* dalla classe giusta (0 = perfetto). **R²** = quanto le probabilità si avvicinano
alle risposte vere rispetto a dare a tutti le frequenze delle classi. Per due classi i numeri sono
calcolati come nelle sezioni KPI. In più: **accuratezza bilanciata** (media della sensibilità di
ogni classe: con 3 classi, tirare a caso dà 0,33) e **AUC** media.
"""
