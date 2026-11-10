# Bioscience-PLM

Valutiamo se **ESM-2**, un modello preaddestrato su sequenze proteiche, riesce a prevedere l’ordinamento degli effetti delle mutazioni di TEM-1 usando dati sperimentali pubblicati. Il benchmark confronta 35M, 150M e 650M parametri; la **baseline è 35M su GPU del 10 novembre 2026**.

**Indice:** [Biologia](#1-biologia-e-domanda-scientifica) · [Dati](#2-dati-e-punteggio-sperimentale) · [Metodo](#3-predizione-zero-shot) · [Risultati](#4-risultati-e-grafici) · [Bootstrap](#5-bootstrap-per-posizione) · [Notebook e sviluppi](#6-notebook-e-futuro-esperimento-supervisionato) · [Uso e riproducibilità](#7-installazione-esecuzione-e-riproducibilità)

## 1. Biologia e domanda scientifica

Una proteina è una catena di amminoacidi, detti anche *residui*. **TEM-1 è una beta-lattamasi**, cioè un enzima capace di inattivare antibiotici beta-lattamici come l’ampicillina. Una TEM-1 sufficientemente funzionante può proteggere il batterio **Escherichia coli** che la produce.

Il **wild type (WT)** è la sequenza di riferimento. Una *sostituzione amminoacidica* cambia un residuo: per esempio `H24V` indica istidina (`H`) sostituita da valina (`V`) alla posizione 24. La sequenza risultante è una variante.

Qui **fitness** significa prestazione del batterio sotto antibiotico: maggiore fitness indica che se la cava meglio nelle condizioni del saggio. Una mutazione che compromette la protezione può aumentare la sensibilità all’ampicillina e ridurre la crescita. Tuttavia, la crescita è una misura complessiva: non identifica automaticamente cambiamenti nella catalisi, nella stabilità, nella quantità della proteina o in altri fattori.

Fitness, patogenicità (capacità di causare malattia) e capacità dell’antibiotico di uccidere **non sono misure intercambiabili**. La finalità applicativa è comprendere fenomeni utili a contrastare la resistenza; l’obiettivo attuale è migliorare la **previsione degli effetti in entrambe le direzioni**, non massimizzare la fitness batterica.

## 2. Dati e punteggio sperimentale

Il **deep mutational scanning (DMS)** misura molte varianti in parallelo. La selezione modifica la loro rappresentazione nella popolazione; conteggi ottenuti mediante sequenziamento permettono di stimarne l’arricchimento o l’impoverimento relativo. Noi analizziamo dati già pubblicati, senza eseguire esperimenti biologici.

**[ProteinGym](https://github.com/OATML-Markslab/ProteinGym#overview)** raccoglie e standardizza questi saggi. `BLAT_ECOLX_Stiffler_2015` contiene **4.996 sostituzioni singole**, nelle **263 posizioni 24–286** di un riferimento lungo **286 residui**. La condizione selezionata è **2.500 µg/mL di ampicillina**, non una media delle concentrazioni. Delle 4.997 sostituzioni possibili nella regione manca `H24A`; non imputiamo un valore. Le coordinate partono da 1 nella sequenza ProteinGym, senza offset aggiuntivi.

| Colonna del CSV | Significato |
| --- | --- |
| `mutant` | Sostituzione, per esempio `H24V` |
| `mutated_sequence` | Sequenza completa della variante |
| `DMS_score` | Misura sperimentale continua, target della valutazione |
| `DMS_score_bin` | Classe ottenuta confrontando il punteggio con una soglia |

Per [Stiffler et al., equazione 1](https://ranganathanlab.org/wp-content/uploads/2018/03/Stiffler_Ranganathan.pdf), l’arricchimento confronta conteggi sotto selezione e senza selezione. Il punteggio è:

```text
F = log10(arricchimento della variante / arricchimento del WT)
```

In parole: si confronta la prestazione relativa della variante con quella del riferimento su scala logaritmica. **Vicino a zero significa simile al WT; negativo inferiore; positivo superiore.** Zero non significa assenza di crescita e il punteggio non è una percentuale di sopravvivenza. Un piccolo valore positivo non dimostra da solo un vantaggio significativo.

ProteinGym orienta gli score verso fitness crescente, ma lo zero non è universale fra saggi. Qui `DMS_score_bin` usa la **mediana −1,159498916**: classe 1 significa sopra tale soglia, non necessariamente sopra il WT. Non è una soglia clinica di resistenza; il benchmark usa lo score continuo.

## 3. Predizione zero-shot

Un **protein language model (pLM)** apprende regolarità delle sequenze proteiche. ESM-2 può stimare quali amminoacidi siano plausibili in un contesto. Mascheriamo una sola posizione del WT e confrontiamo, nello stesso contesto, la probabilità dell’alternativa con quella del residuo originale:

```text
esm2_score = ln P(alternativa | WT mascherato) − ln P(originale | WT mascherato)
```

`ln` è il logaritmo naturale. Per `H24V`, un valore positivo significa che il modello preferisce V a H; uno negativo indica la preferenza opposta. **Non dimostra automaticamente perdita di funzione.** `DMS_score` è un risultato sperimentale; `esm2_score` è una preferenza probabilistica: scale e zeri hanno significati diversi.

L’ipotesi è: **le sostituzioni meno compatibili con le regolarità apprese dal modello tendono anche a compromettere maggiormente la prestazione misurata?** Il modello non riceve la concentrazione dell’antibiotico.

**Zero-shot** significa nessun addestramento sulle etichette DMS di questo saggio; proteina o sequenze omologhe, cioè evolutivamente imparentate, potrebbero essere presenti nel pretraining.

Il metodo conserva lo scoring originale, detto *masked marginals*: una distribuzione per posizione viene riutilizzata per tutte le sue sostituzioni. Si mantiene l’intero WT come contesto, senza troncature; inferenza senza gradienti, FP32 e TF32 disabilitato. Dati incoerenti vengono rifiutati, non filtrati silenziosamente. Il benchmark aggiunge esecuzione GPU, confronto delle tre dimensioni e valutazione statistica appaiata.

## 4. Risultati e grafici

La **correlazione di Spearman (ρ)** confronta le graduatorie sperimentale e predetta: +1 indica ordine concorde, −1 opposto, circa 0 scarso accordo monotono. **0,73 non significa 73% di predizioni corrette**, né misura la precisione dei valori assoluti.

Run completi del **10 novembre 2026**, RTX 3090, stesse 4.996 varianti:

| ESM-2 | Spearman ρ | Intervallo bootstrap 95% | Inferenza | Batch |
| --- | ---: | --- | ---: | ---: |
| **35M, baseline** | **0,556712** | [0,502; 0,609] | 0,665 s | 8 |
| 150M | 0,660518 | [0,613; 0,703] | 2,283 s | 8 |
| 650M | 0,731491 | [0,683; 0,771] | 7,116 s | 4 |

M significa milioni di parametri. La GPU accelera il calcolo; il miglioramento di correlazione osservato emerge dal confronto **fra modelli**. Una dimensione maggiore non garantisce risultati migliori su qualsiasi proteina. Tempi esclusi caricamento, download, warmup e grafici; batch differenti, quindi non è un confronto ottimizzato del throughput.

Nei [grafici 650M](results/scale_gpu/650m/correlation.png), il pannello degli score confronta DMS sull’asse orizzontale ed ESM sul verticale: interessa la tendenza crescente, non l’uguaglianza numerica. Il pannello dei ranghi mostra direttamente l’accordo fra ordinamenti. Nel [profilo per posizione](results/scale_gpu/650m/position_profile.png), ogni punto è la media delle sostituzioni misurate: evidenzia sensibilità o tolleranza locali, senza identificarne il meccanismo biologico; la copertura può differire.

## 5. Bootstrap per posizione

Le sostituzioni dello stesso sito condividono contesto e vincoli biologici: non le trattiamo come osservazioni indipendenti. Il **bootstrap** ricampiona blocchi di dati per valutare quanto il risultato dipenda dalla composizione delle posizioni.

Eseguiamo **2.000 ripetizioni**, seed 0. In ciascuna estraiamo **263 posizioni con ripetizione**, mantenendo tutte le sostituzioni di ogni posizione estratta. Il confronto è **appaiato**: tutti i modelli vengono valutati sulle medesime estrazioni. Non riaddestriamo modelli e non ricalcoliamo predizioni; cambiano posizioni incluse e loro peso.

Esempio verificato sulle predizioni salvate:

| Posizioni considerate | ρ 35M | ρ 650M |
| --- | ---: | ---: |
| 100, 101, 102, 103 | 0,776 | 0,817 |
| 102, 100, 102, 103 | 0,805 | 0,871 |
| 101, 101, 103, 103 | 0,630 | 0,558 |

Sono **sottoinsiemi illustrativi**, non risultati completi. Ognuna di queste posizioni contiene 19 sostituzioni: ripetere 102 ripete l’intero blocco, quindi ogni riga valuta 76 osservazioni, eventualmente duplicate.

Calcoliamo ρ e la differenza fra modelli in ogni ripetizione. I **percentili 2,5 e 97,5** delimitano l’intervallo centrale del 95% dei valori ricampionati. Sul benchmark completo, 650M ha ρ ≈ **0,731**, intervallo **[0,683; 0,771]**; la differenza **650M − 35M ≈ +0,175**, intervallo **[+0,125; +0,223]**. Quest’ultimo è interamente positivo e sostiene il vantaggio nel ricampionamento adottato.

La correlazione sull’intero dataset resta un valore preciso. L’intervallo descrive variabilità interna al saggio rispetto alla composizione delle posizioni: **non** l’errore di ogni variante, le repliche di laboratorio o una garanzia di generalizzazione ad altre proteine.

## 6. Notebook e futuro esperimento supervisionato

[tem1_exploration.ipynb](tem1_exploration.ipynb) conserva l’esplorazione iniziale: estrae **embedding**, rappresentazioni numeriche del WT, con ESM-2 150M ed esplora i DMS. Non contiene un predittore supervisionato delle varianti. Avviarlo dalla radice della repository: i percorsi relativi di FASTA e ZIP restano validi. Il notebook scarica il FASTA e usa CPU; non sostituisce la pipeline riproducibile.

Il futuro esperimento supervisionato, **non ancora implementato**, avrà logica in moduli Python riutilizzabili ed eventualmente un notebook per esplorazione e presentazione. Prima baseline: pLM **congelato**, senza aggiornare i suoi parametri, e regressore semplice addestrato sui DMS. Servono rappresentazioni distinte, per esempio embedding delle sequenze mutate o differenze rispetto al WT: un unico vettore globale WT identico per tutte le varianti non basta.

Training e test saranno separati **per posizione**; gli iperparametri verranno scelti con validazione interna al training, senza usare il test. Confronteremo regressore e zero-shot sulle **stesse posizioni di test**. Generalizzare ad altre proteine richiederà ulteriori saggi e controllo dell’omologia.

## 7. Installazione, esecuzione e riproducibilità

`src/tem1/` contiene dati, scoring, valutazione e CLI; `benchmark.py` esegue la suite, `compare.py` il bootstrap; `tests/` verifica la correttezza. `docs/` conserva soltanto il [riepilogo JSON](docs/scaling-summary.json).

Dalla radice `Bioscience-PLM`, creare l’ambiente solo se assente:

```bash
uv --no-config venv --python 3.12
uv --no-config pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1'
uv --no-config pip install --python .venv/bin/python -e . -c requirements-baseline.txt
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Python supportato: 3.10–3.13. `--no-config` evita l’incompatibilità locale fra uv 0.9.28 e la configurazione di sistema. I vincoli diretti non sono un lockfile completo. Per il notebook:

```bash
uv --no-config pip install --python .venv/bin/python -c requirements-baseline.txt biopython pandas requests jupyter
jupyter notebook tem1_exploration.ipynb
```

Dati già presenti localmente; in un clone nuovo, il download trasferisce l’archivio ProteinGym v1.3 completo conservando solo il saggio necessario:

```bash
tem1 download
tem1 inspect
# Alternativa: riusare lo ZIP locale in una directory nuova; scarica solo i metadati.
tem1 download --data-dir data/imported --archive proteingym_substitutions.zip
tem1 inspect --data-dir data/imported
```

La validazione controlla checksum, conteggi, duplicati, score finiti e corrispondenza fra mutazioni e sequenze. Per usare l’import alternativo aggiungere `--data-dir data/imported` ai benchmark:

```bash
# Smoke test: verifica ridotta, non risultato del benchmark completo.
python -m tem1.benchmark --sizes 35m --limit-positions 4 --output results/new_smoke
python -m tem1.benchmark --output results/new_scale
python -m tem1.compare results/new_scale/35m results/new_scale/150m results/new_scale/650m \
  --replicates 2000 --output results/new_scale/comparison.json
```

La suite usa CUDA, processi separati e [revisioni immutabili](configs/esm2-scale.json). Directory non vuote non vengono sovrascritte. Il batch conta posizioni mascherate; in caso di memoria insufficiente ridurlo nella configurazione e scegliere un output nuovo. Per inferenza offline su un singolo modello già in cache, `tem1 run` accetta `--local-files-only`, `--model` e `--revision`.

Su Colab scegliere un runtime GPU, entrare nella repository e usare lo stesso ambiente; partire da 35M/150M con batch 2–4. Verificare CUDA e salvare gli output prima della scadenza della sessione.

Le directory dei run [35M](results/scale_gpu/35m/), [150M](results/scale_gpu/150m/) e [650M](results/scale_gpu/650m/) conservano predizioni, metriche, grafici, configurazioni, revisioni, versioni software e hash del codice. Il [manifest](data/tem1/manifest.json) registra provenienza e checksum dei dati. `data/`, `results/`, cache e ambiente sono esclusi da Git: i link locali richiedono gli artefatti o la loro rigenerazione. Gli artefatti storici restano conservati. Il [tentativo 3B](results/scale_gpu/3b/failure.json) non ha prodotto predizioni perché mancavano pesi `safetensors`; è escluso dalla suite.
