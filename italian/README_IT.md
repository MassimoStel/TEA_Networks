# TEA_Networks — adattamento italiano

Adattamento della libreria [TEA_Networks](https://github.com/MassimoStel/TEA_Networks)
(estrazione di reti Agente–Evento–Target / SVO da testo) alla **lingua italiana**.

La libreria originale è **solo inglese**: il modulo `teanets.svo_extraction` è
scritto sullo schema di dipendenze inglese (ClearNLP: `dobj`, `pobj`, `prep`,
`agent`, `attr`, `acomp`, `oprd`, `dative`, `npadvmod`) e sui tag `VBN`/`VB`,
calcola la valenza dei nodi con VADER (solo EN) e risolve la sinonimia con la
WordNet inglese. Questo modulo re-implementa la **stessa semantica** (chi fa
cosa a chi, con la politica TEA sul passivo) sopra al modello italiano
`it_core_news_lg`, che usa lo schema **Universal Dependencies**.

## File

| File | Contenuto |
|------|-----------|
| `tea_italian.py` | estrattore Agente/Evento/Target italiano (UD), valenza IT, sinonimi IT |
| `prodrop_coref.py` | resolver morfologico del pro-drop (soggetto sottinteso) |
| `coref_model.py` + `train_coref.py` | modello di coreferenza addestrato su CorefUD (spagnolo→italiano) |
| `coref_es_model.joblib` | modello mention-ranking addestrato (975 B) |
| `validate_coref_transfer.py` | transfer del modello addestrato all'italiano |
| `gold_standard_svo_it.csv` | gold standard parallelo (122 frasi, traduzione del gold EN) |
| `gold_realtext_it.csv` | gold testo reale — narrativo (Pinocchio cap. 2, 20 frasi) |
| `gold_wiki_it.csv` | gold testo reale — enciclopedico (Wikipedia: Levi-Montalcini, Turing) |
| `gold_firstperson_it.csv` | gold testo reale — prima persona (Svevo, *La coscienza di Zeno*) |
| `prodrop_it.csv` | sottoinsieme sintetico per misurare l'effetto pro-drop |
| `validate_it.py` | validazione P/R/F1 + accuratezza passivo (gold sintetico) |
| `validate_realtext.py` | validazione su testo reale, baseline vs resolver pro-drop (qualsiasi gold) |
| `VALIDATION_REPORT_IT.md` | report completo di riproduzione e validazione |

## Uso

```python
import sys; sys.path.insert(0, "italian")
from tea_italian import extract_tea_from_text_it, extract_main_triple_it, get_it_nlp

# grafo TEA completo (stesso schema di teanets.extract_svos)
df = extract_tea_from_text_it("Il dottore fu ringraziato dalla famiglia.")

# tripla singola della frase (per validazione)
agente, evento, target, passive_approx, is_passive = \
    extract_main_triple_it(get_it_nlp()("Il topo fu inseguito dal gatto."))
# -> ('gatto', 'inseguire', 'topo', 0, 1)
```

Grafo TEO completo **con recupero del soggetto pro-drop** integrato — basta
`resolve_prodrop=True` (le teste di clausola sono processate in ordine di testo,
la continuità del soggetto attraversa le frasi):

```python
from tea_italian import extract_tea_from_text_it
df = extract_tea_from_text_it(
    "Geppetto prese il legno. Lo portò a casa e cominciò a lavorare.",
    resolve_prodrop=True)
# senza il flag: 1 arco Agente (geppetto->prendere)
# con il flag:   ogni verbo pro-drop è attribuito a 'geppetto'
```

Per uno stream frase-per-frase con stato di discorso condiviso:

```python
from prodrop_coref import ProDropResolver, extract_main_triple_resolved
resolver = ProDropResolver()
for frase in testo_in_ordine:
    agente, evento, target, pa, isp = extract_main_triple_resolved(get_it_nlp()(frase), resolver)
```

Validazione:

```bash
python italian/validate_it.py                                  # gold sintetico (122 frasi)
python italian/validate_realtext.py italian/gold_realtext_it.csv     # narrativo
python italian/validate_realtext.py italian/gold_wiki_it.csv         # enciclopedico
python italian/validate_realtext.py italian/gold_firstperson_it.csv  # prima persona
```

## Mappatura ClearNLP (EN) → Universal Dependencies (IT)

| Ruolo | Inglese (originale) | Italiano (questo modulo) |
|------|---------------------|--------------------------|
| Oggetto diretto | `dobj` | `obj` |
| Oggetto preposizionale | `prep` + `pobj` | `obl` + `case` |
| Soggetto passivo (paziente) | `nsubjpass` | `nsubj:pass` |
| Ausiliare passivo | `auxpass` | `aux:pass` (essere / venire) |
| Agente ("by / da") | `agent` + `pobj` | `obl:agent` |
| Copula | (predicato = `attr`/`acomp`) | `cop` (predicato = ROOT) |
| Passivante impersonale | — | `expl:impers` (si-passivante) |

## Risorse linguistiche

- **Valenza**: lessico italiano `teanets.valence.italian` (2001 positivi, 2708
  negativi), con inversione per negazione ("non ...").
- **Sinonimia**: WordNet italiana via Open Multilingual WordNet
  (`nltk` `wordnet.synsets(..., lang='ita')`).

## Limiti noti (documentati e quantificati)

1. **Coreferenza neurale**: fastcoref/stanza/maverick sono solo inglesi; nessun
   modello CorefUD italiano con *zero-mention* è disponibile come pacchetto.
2. **Pro-drop**: l'italiano omette il soggetto pronominale → nessun `nsubj` da
   estrarre. È **mitigato** dal resolver morfologico integrato in
   `extract_tea_it(..., resolve_prodrop=True)`. Su testo reale il recupero del
   soggetto sottinteso passa da **0/18 a 13/18** su tre generi (narrativo 7/8,
   enciclopedico 4/6, prima persona 2/4) e l'Agente F1 sale in tutti i generi
   (es. narrativo 0.645→0.895). La 1ª/2ª persona si risolve per pura morfologia;
   per la 3ª con soggetti interposti/alternati servirebbe un modello CorefUD con
   zero-mention.
3. **Qualità del modello `it_core_news_lg`**: non esiste una variante
   transformer ufficiale per l'italiano. Il tetto delle prestazioni è fissato
   dagli errori del lemmatizzatore sul passato remoto (`ruppe`, `scosse`,
   `decisero`) e dagli omografi verbo/nome (`legge`, `bolle`, `brillano`),
   non dalla logica di estrazione.
