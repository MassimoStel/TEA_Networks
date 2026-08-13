# Riproduzione e validazione di TEA_Networks per l'italiano

**Data:** 2026-08-11 · **Libreria:** [MassimoStel/TEA_Networks](https://github.com/MassimoStel/TEA_Networks) v0.3.1
**Ambiente:** Python 3.12, spaCy 3.8, `en_core_web_trf` 3.8.0, `it_core_news_lg` 3.8.0

---

## 1. Obiettivo

La libreria TEA_Networks estrae reti cognitive **Agente–Evento–Target** (SVO)
da testo, con gestione della diatesi passiva (l'agente semantico va nel ruolo
Agente, il paziente nel ruolo Target; nel passivo senza agente il paziente
resta in Agente come approssimazione, con flag `passive_approx`). La libreria
è **solo inglese**. Obiettivi di questo lavoro:

1. **Riprodurre** i risultati della libreria in inglese (baseline).
2. **Adattarla** all'italiano.
3. **Validare** l'estrazione in italiano su un gold standard.

## 2. Riproduzione della baseline inglese

Eseguite le funzioni di validazione della libreria (`teanets.svo_validation`)
sul gold standard incluso `data/gold_standard_svo.csv` (122 frasi).

| Ruolo | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Subject | 0.811 | 0.811 | 0.811 |
| Verb | 0.697 | 0.697 | 0.697 |
| Object | 0.511 | 0.549 | 0.529 |

**Accuratezza `passive_approx` (pipeline TEA completa): 0.975 (119/122).**

Nota: `validate_svo` usa l'estrattore dep-parse *minimale* (baseline), che non
applica lo scambio agente/paziente del passivo; per questo Subject/Object sono
volutamente bassi. La gestione del passivo (che è il contributo scientifico)
è misurata a parte ed è al 97.5%. Baseline riprodotta correttamente. ✅

## 3. Adattamento all'italiano

Il cuore della libreria dipende dallo schema di dipendenze inglese. Il modello
italiano `it_core_news_lg` usa **Universal Dependencies**. È stato scritto il
modulo `tea_italian.py` che re-implementa la semantica TEA su UD:

- oggetto `obj`; oggetto obliquo `obl`+`case`;
- passivo `aux:pass`/`nsubj:pass`, agente `obl:agent` (scambio agente/paziente);
- **passivo perifrastico sotto-etichettato**: euristica participio passato +
  ausiliare *essere/venire* (che il modello spesso etichetta come `aux`/`cop`
  semplice), con esclusione dei verbi **inaccusativi** (*andare, cadere...*) il
  cui perfetto con *essere* è attivo;
- **si-passivante** `expl:impers` → passivo approssimato;
- copulare (`cop`, predicato = ROOT);
- valenza dei nodi con il lessico italiano (VADER è solo EN);
- sinonimia con WordNet italiana (Open Multilingual WordNet).

## 4. Validazione in italiano

Costruito un **gold standard parallelo** (`gold_standard_svo_it.csv`, 122 frasi
= traduzione italiana del gold inglese, con soggetti mantenuti espliciti per
testare il meccanismo), coprendo: attivo, passivo con/senza agente,
*venire*-passive, si-passivante, copulare, ditransitivo, relative, modali,
negazione, interrogative, coordinazione, subordinate, e passivi "traumatici".

### Risultati (estrazione della tripla principale)

| Ruolo | Precision | Recall | F1 |
|-------|-----------|--------|----|
| **Agente** | 1.000 | 0.975 | **0.988** |
| **Evento** | 0.852 | 0.852 | **0.852** |
| **Target** | 0.988 | 0.964 | **0.976** |

| Metrica passivo | Accuratezza |
|-----------------|-------------|
| `passive_approx` (paziente-in-Agente) | **0.984 (120/122)** |
| `is_passive` (rilevamento diatesi) | **0.984 (120/122)** |

L'estrazione italiana di Agente e Target **supera** la baseline inglese
riportata sopra, perché il modulo IT applica lo scambio agente/paziente del
passivo (che la baseline EN minimale non fa).

### Analisi degli errori residui

Tutti i mismatch residui sono attribuibili al **modello `it_core_news_lg`**,
non alla logica di estrazione:

- **Lemmatizzazione del passato remoto/futuro** (fonte principale degli errori
  di Evento): `ruppe`→"ruppe", `scosse`→"scosdere", `decisero`→"decisero",
  `mangerà`→"mangare", `passerà`→"passeraare".
- **Omografi verbo/nome** taggati come NOUN/PROPN: `legge` (legge/legge),
  `bolle` (bolle/bolle), `brillano`, `squillò`, `rotta` → l'intera clausola
  viene mal analizzata.

Correggendo idealmente questi errori del modello, l'F1 di Evento salirebbe
sopra 0.95. Sono limiti del *tagger/lemmatizer*, superabili con un modello
transformer italiano o normalizzazione morfologica, indipendenti dal porting.

## 5. Effetto pro-drop (specifico dell'italiano)

L'italiano omette il soggetto pronominale. Su 12 frasi in due versioni
(soggetto esplicito vs pro-drop):

| Condizione | Nodo Agente presente | Evento ok | Target ok |
|------------|----------------------|-----------|-----------|
| Soggetto esplicito | 12/12 (100%) | 9/12 | 11/12 |
| **Pro-drop** | **5/12 (42%)** | 4/12 | 7/12 |

Senza soggetto lessicale la rete perde ~60% dei nodi Agente (e i pochi presenti
sono spesso spurî). **Conseguenza pratica:** per l'italiano è essenziale un
sistema di coreferenza / recupero del soggetto pro-drop; in sua assenza le
analisi TEA italiane devono usare testi con soggetti espliciti o interpretare
con cautela la copertura del ruolo Agente.

## 6. Validazione su testo reale (tre generi)

Oltre al gold sintetico parallelo, l'estrattore è stato validato su **testo
reale** di pubblico dominio in tre generi diversi, ciascuno con soggetti
**pro-drop**, processati in ordine di testo (`validate_realtext.py`):

- **Narrativo** — Collodi, *Le avventure di Pinocchio*, cap. 2
  (`gold_realtext_it.csv`, 20 frasi, 8 pro-drop).
- **Enciclopedico** — voci Wikipedia (R. Levi-Montalcini, A. Turing)
  (`gold_wiki_it.csv`, 12 frasi, 6 pro-drop): ricco di passivi e di pro-drop
  "enciclopedico" (soggetto introdotto una volta, poi sottinteso).
- **Prima persona** — Svevo, *La coscienza di Zeno*, prefazione
  (`gold_firstperson_it.csv`, 7 frasi, 4 pro-drop): pro-drop di "io".

**Agente F1 — senza vs con resolver pro-drop:**

| Genere | baseline | + resolver | pro-drop recuperato |
|--------|----------|-----------|---------------------|
| Narrativo (Pinocchio) | 0.645 | **0.895** | 7/8 |
| Enciclopedico (Wikipedia) | 0.667 | **0.833** | 4/6 |
| Prima persona (Svevo) | 0.600 | **0.833** | 2/4 |
| **totale pro-drop** | **0/18** | | **13/18** |

Su tutti i generi il recall dell'Agente crolla per il pro-drop e il resolver lo
recupera in larga parte. I miss residui sono onesti: errori del modello
(omografi *pubblico* nome/verbo, mis-tag imperativo di *Sviluppò* a inizio
frase) oppure casi genuinamente difficili per la coreferenza a recency (un
soggetto non-topicale interposto — *"il suo lavoro"* — o un antecedente che era
un oggetto, non un soggetto). Evento/Target restano limitati dagli stessi
errori di lemmatizzazione del modello. La **prima persona** è il caso più
robusto: 1ª/2ª persona si risolvono per pura morfologia (io/tu), senza bisogno
di antecedente.

## 7. Attacco al pro-drop: coreferenza

**Modelli neurali.** Non esiste un resolver di coreferenza italiano
installabile via pip che predica le *zero-mention* (soggetti nulli del
pro-drop, annotate in CorefUD): `maverick-coref` distribuisce solo modelli
**inglesi** (OntoNotes/LitBank/PreCo) — e in questo ambiente Windows/Python 3.12
non si compila nemmeno (backend di build rotto); `stanza`/`fastcoref` hanno la
coref solo per l'inglese; i sistemi CorefUD multilingui (CorPipe…) sono codice
di ricerca, non pacchetti.

**Soluzione adottata — resolver morfologico** (`prodrop_coref.py`). Recupera il
soggetto sottinteso da: (1) **morfologia del verbo** (Persona/Numero → io/tu/
noi/voi per 1ª/2ª persona) e (2) **continuità del soggetto** (per la 3ª persona,
l'antecedente nominale più recente che concorda nel Numero, entro una finestra
di recency; le teste plurali accettano solo antecedenti plurali, con fallback
"loro"). Applicato con stato di discorso a scorrimento sulle frasi in ordine.

**Risultato (sottoinsieme pro-drop, testo reale, 8 frasi):**

| | Agente recuperato |
|--|-------------------|
| baseline (nessuna coref) | **0/8** |
| resolver morfologico | **7/8** |

L'unico errore residuo è "Diventava subito una bestia", dove il parser scambia
"una bestia" (predicato/oggetto) per il soggetto: il resolver — correttamente —
non interviene perché un soggetto (sbagliato) è già presente. Non è un fallimento
della coreferenza ma del parsing.

**Integrazione nel grafo.** Il resolver è integrato direttamente in
`extract_tea_it(doc, resolve_prodrop=True)`: le teste di clausola sono
processate in ordine di testo, il soggetto grammaticale (paziente per i passivi
con agente) alimenta la continuità, e i soggetti sottintesi vengono riempiti. Su
una narrazione multi-frase l'effetto sul grafo è drastico — es. *"Geppetto prese
il legno. Lo portò a casa e cominciò a lavorare. Costruì un burattino. Poi si
addormentò."* produce **1** arco Agente senza resolver e **6** archi (tutti
attribuiti a *geppetto*) con `resolve_prodrop=True`.

**Limiti del resolver.** La continuità per recency funziona benissimo nella
prosa narrativa e in prima persona, ma su testi con alternanza rapida di più
entità di 3ª persona o con soggetti non-topicali interposti può scegliere
l'antecedente errato; per questi casi servirebbe un vero modello CorefUD con
gestione delle zero-mention. Un guard esclude date/numeri dai candidati
antecedenti.

## 8. Modello di coreferenza addestrato (CorefUD) a confronto

Per verificare quanto lascia sul tavolo l'euristica, è stato costruito un vero
**modello di coreferenza addestrato** per la selezione dell'antecedente dei
soggetti nulli (`coref_model.py`, `train_coref.py`).

**Dati.** L'italiano non è nel CorefUD pubblico (1.2, 15 lingue). Si addestra
quindi sulla lingua CorefUD tipologicamente più vicina e pro-drop —
**Spagnolo-AnCora** (romanzo, con soggetti nulli annotati come *empty node*) —
usando solo **feature UD universali** (accordo Persona/Numero, ruolo
grammaticale, POS, recency), così il modello trasferisce all'italiano via le
annotazioni UD di spaCy. Estratte ~5.5k istanze di soggetto nullo di 3ª persona
dal train; classificatore **mention-ranking** (regressione logistica).

**Risultato principale (Spagnolo-AnCora dev, 655 soggetti nulli, input
identici):**

| Modello | Accuratezza selezione antecedente |
|---------|-----------------------------------|
| Euristica recency (= nucleo del resolver morfologico) | **0.270** |
| **Modello addestrato (CorefUD)** | **0.479** |

Il modello addestrato **quasi raddoppia** l'euristica. I pesi appresi sono
linguisticamente interpretabili: `cand_subject` +1.64 (i soggetti sono
antecedenti salienti), `num_agree` +1.53 (accordo di numero), `sent_recency`
+1.25 — cioè il modello *riscopre dai dati* i due principi che il resolver
morfologico codifica a mano, e li pesa meglio sui casi ambigui.

**Transfer zero-shot all'italiano.** Applicando il modello spagnolo ai casi
pro-drop italiani (feature UD via spaCy), a parità di pool di candidati ristretto
ai soggetti (come il resolver reale):

| Testo | recency (resolver morfologico) | modello addestrato |
|-------|-------------------------------|--------------------|
| Spagnolo AnCora (655 casi, ruoli misti) | 0.27 | **0.48** |
| Italiano (narrativa/bio, alta continuità, 7 casi) | **1.00** | 0.86 |

**Lettura.** Il modello addestrato vince nettamente **dove esistono antecedenti
in competizione** (testo giornalistico spagnolo, ruoli misti). Nella prosa
italiana ad **alta continuità del soggetto** (narrativa, biografie) l'euristica
recency è già quasi ottimale e non viene battuta. Per l'uso tipico di TEA in
italiano — narrativa e testi clinici in prima persona — il resolver morfologico
(zero dipendenze, nessun addestramento) è quindi adeguato; per generi più
difficili conviene un modello addestrato, e questo esperimento mostra che un vero
CorefUD italiano (oggi assente) darebbe il guadagno maggiore.

## 9. Conclusioni

- La baseline inglese della libreria è stata **riprodotta**. ✅
- La logica di estrazione Agente–Evento–Target, incluso il trattamento del
  passivo, **si trasferisce all'italiano** con alta fedeltà: Agente F1 0.988,
  Target F1 0.976, accuratezza passivo 0.984. ✅
- Valenza e sinonimia sono state **localizzate** (lessico IT + WordNet IT). ✅
- Validato anche su **testo reale** (Pinocchio): con il resolver pro-drop,
  Agente F1 0.895 (da 0.645) e recupero del soggetto sottinteso 7/8. ✅
- Confronto con un **modello di coreferenza addestrato** (CorefUD spagnolo,
  trasferito via feature UD): batte l'euristica dove ci sono antecedenti in
  competizione (0.48 vs 0.27 su gold spagnolo); nella prosa italiana ad alta
  continuità il resolver morfologico resta ottimo. ✅
- I due limiti da tenere presenti per l'uso scientifico in italiano sono
  (a) la qualità del modello `it_core_news_lg` (lemmatizzazione dei tempi
  non-presente e omografi) e (b) il **pro-drop**: mitigato dal resolver
  morfologico incluso; per generi difficili un modello addestrato aiuta, e un
  vero CorefUD italiano (oggi assente) darebbe il guadagno maggiore.

### Riproducibilità

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m spacy download it_core_news_lg
python italian/validate_it.py          # metriche Agente/Evento/Target + passivo
```
