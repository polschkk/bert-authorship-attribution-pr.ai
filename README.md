# Authorship Attribution: BERT and ModernBERT feature attribution analysis

This project investigates which textual features drive correct classification by
fine-tuned BERT and ModernBERT authorship classifiers. The goal is to make the
classifiers' decisions more interpretable and to identify feature groups that
could be targeted in future obfuscation experiments.

## Research questions

1. What token-level features (content words, punctuation, stop words) does each classifier rely on?
2. Are LLM-generated texts distinguishable from human-written texts at the feature level, and what features differentiate them?
3. Can a reader identify the author or source class just by reading the text?
4. Which features could be targeted for future obfuscation strategies?
5. Does ModernBERT rely on different features than BERT, and why?


## Datasets

### PAN25 (LLM-source attribution)
- **Task:** 23-class classification (22 LLM sources + human)
- **Train:** 23,707 examples | **Val:** 3,556 examples
- **Genres:** fiction, essays, news
- **BERT balanced accuracy (best seed):** 62.3%
- **ModernBERT balanced accuracy (best seed):** 80.2%

### AuthorMix (Author Attribution)
- **Task:** 14-class classification (literary authors, politicians, bloggers, AMT workers)
- **Train:** 14,579 examples | **Val:** 3,642 examples
- **Classes:** Fitzgerald, Hemingway, Woolf, Obama, Bush, Trump, 5 blog authors, 3 AMT workers
- **BERT balanced accuracy (best seed):** 86.2%
- **ModernBERT balanced accuracy (best seed):** 88.2%

## Models

### BERT
- Model: `bert-base-uncased`
- Final PAN25 multiclass and AuthorMix experiments use `max_length=512`
- BERT does not expose newline tokens in the attribution ratios used here; the newline audit is 0.000 for both datasets

### ModernBERT
- Model: `answerdotai/ModernBERT-base` (149M parameters)
- Native context length: 8,192 tokens
- Training and attribution cap: 1,500 tokens (used for memory control and covers more than 95% of PAN25 examples)
- ModernBERT uses byte-level tokenisation. Newline tokens are therefore preserved and must be treated separately before lexical ratios are compared with BERT


## Project structure

```
.
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── pan25_eda.ipynb                          # EDA - PAN25 dataset
│   ├── authormix_eda.ipynb                      # EDA - AuthorMix dataset
│   │
│   ├── baseline_bert.ipynb                      # BERT fine-tuning and evaluation
│   ├── attribution_bert.ipynb                   # BERT attribution (IG, SHAP, AR)
│   ├── attribution_analysis_bert.ipynb          # BERT attribution analysis
│   │
│   ├── baseline_modernbert.ipynb                # ModernBERT fine-tuning and evaluation
│   ├── attribution_modernbert.ipynb             # ModernBERT attribution (IG, SHAP, AR)
│   └── attribution_analysis_modernbert.ipynb    # ModernBERT attribution analysis
│
├── runs/
│   ├── results/
│   │   ├── balanced_accuracy_summary.csv
│   │   ├── modernbert_vs_bert_summary.csv
│   │   ├── pan25_full_validation_*.csv / .json
│   │   ├── authormix_full_validation_*.csv / .json
│   │   ├── pan25_modernbert_*.csv / .json / .json.gz
│   │   └── authormix_modernbert_*.csv / .json
│   │
│   ├── models/                                  # best model checkpoints and label maps
│   └── plots/                                   # notebook-generated diagnostic figures
│
└── reports/
    ├── REPORT_BERT.md                           # BERT attribution findings
    └── REPORT_MODERNBERT.md                     # ModernBERT attribution findings
```


## Notebooks

### `pan25_eda.ipynb` and `authormix_eda.ipynb`
Exploratory data analysis before training. Covers class distribution, text length
analysis, truncation risk, duplicate detection, and dataset quality checks.
Run independently, no prior steps required.

### `baseline_bert.ipynb` - BERT fine-tuning
Multi-seed training of `bert-base-uncased` on both datasets.
- Tokenization with max length 512 for the final multiclass experiments
- 3 random seeds, best seed selected by macro F1
- Saves per-seed metrics, best model checkpoints, and label maps
- Computes confusion matrices and per-class accuracy

### `baseline_modernbert.ipynb` - ModernBERT fine-tuning
Same pipeline as `baseline_bert.ipynb` but for `answerdotai/ModernBERT-base`.
- Max sequence length 1,500 in the experiment pipeline
- Eager attention mode required for compatibility
- Mixed precision used where appropriate; attribution is run in fp32 where needed for stable gradients

### `attribution_bert.ipynb` - BERT attribution
Runs three attribution methods on the full validation sets using the best BERT model.
- **Integrated Gradients (IG):** gradient-based, 15 interpolation steps
- **GradientSHAP:** gradient-based, 10 random baseline samples
- **Attention Rollout (AR):** manual Q/K computation (SDPA workaround)
- Saves raw attribution JSON and ratio CSV per dataset
- Token ratios use four categories internally: content, punctuation, stopword, newline. For BERT, newline ratio is 0.000

### `attribution_modernbert.ipynb` - ModernBERT attribution
Same attribution structure as `attribution_bert.ipynb`, adapted for ModernBERT.
- Sequences truncated at 1,500 tokens for memory reasons
- `internal_batch_size=1` for IG (OOM prevention)
- Manual loop for GradientSHAP (no `internal_batch_size` parameter)
- `attention_mask.expand()` required because ModernBERT does not broadcast mask shapes in the same way
- Byte-level tokens are decoded before ratio computation, so tokens such as `Ġthe` and `Ċ` are handled correctly
- Newline tokens are saved as a separate raw audit category

### `attribution_analysis_bert.ipynb` - BERT attribution analysis
Loads outputs from `attribution_bert.ipynb` and performs the analysis used in the BERT report.
Covers token heatmaps, POS analysis, lexical sophistication, attribution
concentration, cross-method agreement, vocabulary fingerprints, positional
analysis, statistical testing, and summary plots.

### `attribution_analysis_modernbert.ipynb` - ModernBERT attribution analysis
Same analysis pipeline as `attribution_analysis_bert.ipynb`, adapted for ModernBERT.
The notebook keeps newline as an audit category but uses non-newline-normalised
content, punctuation, and stopword ratios for thesis-facing lexical plots and
BERT-vs-ModernBERT comparisons.


## How to run

Run in the following order:

```
# Exploratory (optional)
pan25_eda.ipynb
authormix_eda.ipynb

# BERT pipeline
1. baseline_bert.ipynb                    → trains BERT, saves to runs/models/
2. attribution_bert.ipynb                 → requires step 1
3. attribution_analysis_bert.ipynb        → requires step 2

# ModernBERT pipeline
4. baseline_modernbert.ipynb              → trains ModernBERT, saves to runs/models/
5. attribution_modernbert.ipynb           → requires step 4
6. attribution_analysis_modernbert.ipynb  → requires step 5
```

Dependencies are installed inline in each notebook. For local runs:

```bash
pip install -r requirements.txt
```


## Key findings

### Performance

| Dataset | BERT (mean ± std) | ModernBERT (mean ± std) | Difference |
|---------|:-----------------:|:-----------------------:|:----------:|
| PAN25 (23 classes) | 0.615 ± 0.009 | 0.797 ± 0.007 | +18.1 pp |
| AuthorMix (14 classes) | 0.857 ± 0.005 | 0.878 ± 0.004 | +2.1 pp |

### Token-ratio correction

ModernBERT exposes newline tokens, while BERT does not expose them in the same
way. The ratio CSVs therefore keep four raw categories:

```
content, punctuation, stopword, newline
```

For direct BERT-vs-ModernBERT lexical comparisons, the thesis-facing plots use
non-newline-normalised ratios:

```
content / (content + punctuation + stopword)
punctuation / (content + punctuation + stopword)
stopword / (content + punctuation + stopword)
```

The raw newline ratio is kept as an audit signal. It is not used as a main graph
category in the thesis.

### What features drive classification?

**BERT - PAN25 (LLM-source attribution):**
- Relies strongly on stopword and punctuation patterns
- Human text has low IG content ratio (0.215) and high IG stopword ratio (0.451)
- Attention Rollout is dominated by punctuation on PAN25 (roughly 95-99% punctuation)
- The signal is strongly front-loaded because BERT only sees up to 512 tokens

**ModernBERT - PAN25 (LLM-source attribution):**
- Has a strong raw newline signal on PAN25: IG mean 0.333, SHAP mean 0.315, AR mean 0.210
- After newline correction, the direct content-ratio shift relative to BERT is small
- Non-newline IG content ratio changes from 0.237 in BERT to 0.254 in ModernBERT on PAN25
- Punctuation remains the largest non-newline category for many PAN25 classes
- ModernBERT improves PAN25 accuracy strongly, but the gain is not explained by content ratio alone

**Both models - AuthorMix (author attribution):**
- AuthorMix is more content-oriented than PAN25
- Topic-specific nouns and author-specific vocabulary are important
- Some author pairs also show punctuation-style differences
- Attribution is more distributed across the text than in PAN25

### BERT vs ModernBERT interpretation

The corrected result is not that ModernBERT simply changes the task into a
content-word problem. The stronger conclusion is more careful:

- ModernBERT improves classification performance, especially on PAN25.
- ModernBERT exposes formatting/newline information that BERT does not expose in the same way.
- After removing newline tokens from the denominator, the lexical content-ratio difference between BERT and ModernBERT is small.
- The two models still differ in context length, tokenizer, pretraining setup, and attention behaviour, so attribution differences cannot be assigned to architecture alone.

### Obfuscation implications

| Task | BERT target | ModernBERT target |
|------|------------|------------------|
| PAN25 LLM-source attribution | Rewrite the opening; alter function-word and punctuation patterns | Check newline/paragraph formatting; alter opening structure, punctuation, stopword patterns, and class-specific lexical cues |
| AuthorMix author attribution | Replace topic-specific vocabulary across the full text | Replace topic-specific vocabulary across the full text; also check punctuation style for some authors |


## Environment

- Python 3.10+
- Google Colab T4 GPU (16 GB VRAM)
- BERT: `bert-base-uncased`
- ModernBERT: `answerdotai/ModernBERT-base`
