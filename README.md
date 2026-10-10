# Manny

Sistema multi-agente che gira interamente in locale. Dato un compito di programmazione,
un orchestratore crea a runtime gli agenti che servono (ruolo, istruzioni e strumenti di
ciascuno), che scrivono il codice, lo eseguono in una sandbox, leggono gli errori dei test
e li correggono, cercando documentazione sul web quando serve.

Il modello e il contesto vengono scelti in base all'hardware rilevato, e ogni operazione che
consuma memoria passa da un governatore che non lascia intaccare la riserva del sistema.

## Requisiti

- Linux, Python 3.11 o superiore
- [Ollama](https://ollama.com) 0.40 o superiore in esecuzione (le stime di memoria sono
  calibrate su questa versione).
- Docker, per la sandbox e per SearXNG
- Opzionale: GPU NVIDIA con `nvidia-smi` funzionante. Senza, si usano solo modelli piccoli su CPU.

## Installazione

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
ollama pull ornith:9b
.venv/bin/manny doctor
```

`manny doctor` mostra l'hardware rilevato, il budget di memoria, i modelli adatti a questa
macchina (il primo è quello che verrà usato) e i prerequisiti mancanti. Se `ornith:9b` non
entra nella memoria disponibile, indica quale modello scaricare al suo posto.

## Uso

```bash
manny services up        # avvia SearXNG (ricerca web), facoltativo
manny run "Scrivi una libreria per il parsing di file INI, con test"
manny run "Aggiungi la gestione degli errori" -w percorso/del/progetto
manny services down
```

Senza `-w` il lavoro avviene in `workspaces/<data-ora>`. Gli agenti leggono e scrivono solo
dentro quella cartella. Alla fine vengono riportati l'esito, il modello usato e i picchi di
memoria della sessione.

## Come funziona

| Parte | Dove | Cosa fa |
|---|---|---|
| Profilo hardware | `hardware/profiler.py` | Rileva CPU, RAM, VRAM e disco, e assegna una fascia (`cpu-low` ... `gpu-12+`) |
| Governatore | `hardware/governor.py` | Ammette o rifiuta ogni richiesta di memoria; tiene libera una riserva di RAM e un margine di VRAM |
| Catalogo modelli | `data/models.yaml`, `llm/catalog.py` | Stima la memoria di ogni modello e sceglie, tra quelli che entrano nel budget, quello con il punteggio misurato più alto |
| Gestore del modello | `llm/manager.py` | Tiene caricato un solo modello per tutti gli agenti; sotto pressione passa a uno più piccolo |
| Orchestratore | `agents/orchestrator.py` | Chiede al modello un piano (tipo di lavoro, criteri, agenti e compiti), lo valida, lo esegue e ripianifica una volta se un compito fallisce |
| Verifiche e revisione | `checks.py`, `agents/review.py` | Controlli automatici per tipo di lavoro e rilettura finale rispetto ai criteri |
| Agente | `agents/agent.py` | Ciclo pensa, usa uno strumento, osserva. Rileva i blocchi e compatta la cronologia quando il contesto si riempie |
| Sandbox | `sandbox/docker.py` | Un container per task con limiti di RAM, CPU e processi, rete spenta salvo `pip install` |
| Ricerca web | `services.py`, `tools/web.py` | SearXNG locale; le pagine vengono ridotte a testo e gli indirizzi non pubblici sono rifiutati |

Gli agenti sono logici, non processi: condividono lo stesso modello caricato e lavorano uno
alla volta. Si passano risultati sintetici tramite una lavagna condivisa, non le conversazioni.

### Come viene giudicato il risultato

L'orchestratore classifica ogni task e scrive i criteri di accettazione, compreso ciò che
un professionista consegnerebbe anche senza che venga chiesto (gestione degli input errati,
password cifrate, contenuti reali al posto dei segnaposto). Il tipo decide la verifica
automatica:

| Tipo | Verifica automatica |
|---|---|
| `python` | `pytest` nella sandbox |
| `web` | Controllo statico delle pagine: file collegati esistenti, niente segnaposto, testo e stile sufficienti |
| `research` | `report.md` deve esistere e citare solo URL realmente restituiti dagli strumenti web nella sessione |
| `other` | Nessuna: decide la revisione |

Se un agente dichiara di aver finito e la verifica fallisce, riceve l'esito e deve
correggere. A lavoro concluso un revisore rilegge i file rispetto a richiesta e criteri e
ordina i rilievi per gravità. Ogni rilievo va a un correttore dedicato, uno alla volta
(`agent.max_fix_issues`, predefinito 3):

- il correttore non può chiudere senza aver modificato almeno un file;
- se dopo la correzione le verifiche non passano più, la modifica viene annullata e il
  rilievo resta aperto;
- se passano, una domanda mirata al modello conferma che quel rilievo sia davvero risolto.

I rilievi rimasti aperti sono elencati nel risultato. `agent.max_review_rounds: 0` spegne la
revisione.

Se la cartella di lavoro superava già le verifiche all'inizio e la sessione fallisce, i file
vengono riportati allo stato iniziale: un progetto funzionante non viene lasciato rotto.

## Modelli

Il modello viene scelto tra quelli installati che entrano nel budget di memoria, in ordine
di punteggio misurato. Sulla macchina di sviluppo (RTX 2060 da 6 GB, 16 GB di RAM):

| Modello | Punteggio | Velocità | Note |
|---|---|---|---|
| `ornith:9b` | 97 | 15 token/s | Scelta predefinita |
| `qwen2.5-coder:7b` | 55 | 25-30 token/s | Superato da `ornith:9b` |
| North-Mini-Code 30B a 3 bit | 62 (3 compiti su 5) | 18 token/s | Scartato: porta la RAM sotto la riserva |
| Gemma 4 12B Q6_K | 94 (2 compiti su 5) | 4,5 token/s | Scartato qui: per il 60% in RAM, da 5 a 25 volte più lento |
| Gemma 4 12B QAT | non misurato | 8 token/s | Scartato qui: metà della velocità di `ornith:9b` |

I modelli scartati lo sono per questa macchina: con più VRAM il confronto va rifatto.

Il punteggio è la quota di test superati nel banco di prova:

```bash
manny bench ornith:9b qwen2.5-coder:7b     # tutti i compiti
manny bench mio-modello:tag -t bugfix -t roman
manny run "..." --model mio-modello:tag    # usa un modello specifico
```

`manny bench` fa svolgere a ogni modello un set fisso di compiti con l'intero sistema. I
compiti Python sono valutati con test che gli agenti non vedono, quindi un modello non può
passare scrivendosi test deboli. Riporta anche velocità, memoria reale e RAM minima rimasta
libera. Un modello non presente nel catalogo viene dimensionato dai metadati di Ollama; per
farlo scegliere in automatico va aggiunto a `models.yaml` con il punteggio ottenuto.

Su un'altra macchina conviene rilanciare `manny bench`: quale modello rende meglio dipende
da quanta parte ne sta sulla GPU.

## Configurazione

Facoltativa. Viene letto il primo tra `--config`, `$MANNY_CONFIG`, `./manny.yaml` e
`~/.config/manny/config.yaml`. I valori predefiniti sono in `src/manny/config.py`. Esempio:

```yaml
memory:
  ram_reserve_min_mb: 3072     # RAM sempre lasciata al resto del sistema
model:
  name: "qwen2.5-coder:3b"     # forza un modello
  max_ctx: 8192                # meno contesto, circa il 20% più veloce
agent:
  thinking: auto               # ragionamento: auto, always o never
  task_minutes: 10             # tempo massimo per un compito
  run_minutes: 30              # tempo massimo per l'intera sessione
sandbox:
  memory_mb: 2048
```

Per aggiungere modelli si copia `src/manny/data/models.yaml`, lo si estende e lo si indica
in `model.catalog_path`. Le cifre di memoria seguono la contabilità di Ollama (`ollama ps`).

### Ragionamento

Con i modelli che hanno una modalità di ragionamento, `agent.thinking: auto` la usa per
pianificazione, revisione e correttori, e per un agente che continua a ripetere la stessa
azione; nei passi ordinari resta spenta. Misurato con `ornith:9b`: è la combinazione che dà
il punteggio più alto nel minor tempo. `always` raddoppia circa i tempi, `never` è la più
veloce ma i correttori non riescono più a risolvere i rilievi che richiedono un'inferenza.

### Impostazioni consigliate per Ollama

Nel servizio (`sudo systemctl edit ollama`, poi `sudo systemctl restart ollama`):

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="LLAMA_ARG_CACHE_RAM=1024"
```

La seconda limita la cache dei prompt del server dei modelli, che altrimenti cresce fino a
8 GB di RAM durante una sessione. `manny doctor` segnala se manca.

## Limiti attuali

- La sandbox ha solo Python e pytest: il codice in altri linguaggi non viene eseguito.
- La verifica delle ricerche garantisce che le fonti citate siano state consultate, non che le
  affermazioni siano corrette.
- Solo GPU NVIDIA; con altre schede il sistema lavora su CPU.
- Le stime di memoria sono misurate per `ornith:9b`, `qwen2.5-coder:7b` e Gemma 4 12B, ricavate
  dall'architettura per gli altri modelli.
- I punteggi vengono da una sola esecuzione per compito: differenze di pochi punti non sono
  significative.
- Un file lungo costa minuti: a 15 token/s una risposta da 3000 token richiede oltre 3 minuti.
  Le pagine web sono il caso peggiore: nel banco di prova la landing page ha richiesto fino
  a 30 minuti.

## Test

```bash
.venv/bin/python -m pytest
```

I test della sandbox usano Docker reale e vengono saltati se non è disponibile.
