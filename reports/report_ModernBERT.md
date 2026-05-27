# Feature Attribution Analysis Report

## What Features Drive ModernBERT's Authorship Classification?

**Thesis goal:** Identify which text features are responsible for correct
classification by the fine-tuned ModernBERT authorship classifier, and compare
the resulting feature profile with BERT under the same attribution framework.\
**Data:** Full validation sets: PAN25 (3,556 examples, 23 classes) and
AuthorMix (3,642 examples, 14 classes).\
**Methods:** Integrated Gradients (IG), GradientSHAP (SHAP), and Attention
Rollout (AR), all applied to every validation example.\
**Model:** ModernBERT (`answerdotai/ModernBERT-base`) fine-tuned classifiers.
Mean balanced accuracy across three seeds: PAN25 $0.7967 \pm 0.0065$ and
AuthorMix $0.8784 \pm 0.0041$.\
**Token-ratio update:** ModernBERT exposes newline tokens in its tokenizer.
The raw ratios therefore use four categories: content, punctuation, stopword,
and newline. Thesis-facing lexical comparisons use non-newline-normalised
content, punctuation, and stopword ratios. Newline is kept as an audit signal,
not as a plotted thesis category.

## 1. Classification performance context

### PAN25 Multiclass (23 LLM sources + human)

- **Mean balanced accuracy across three seeds:** $0.7967 \pm 0.0065$.
  This is an increase of 18.1 percentage points over the BERT baseline
  ($0.6152$).

- **Best seed:** 0.8024 balanced accuracy.

- **Key point:** The improvement is concentrated on the harder dataset.
  PAN25 requires the classifier to distinguish many similar LLM sources,
  not only human from machine text.

### AuthorMix (14 human authors)

- **Mean balanced accuracy across three seeds:** $0.8784 \pm 0.0041$.
  This is an increase of 2.1 percentage points over the BERT baseline
  ($0.8572$).

- **Best seed:** 0.8818 balanced accuracy.

- **Key point:** ModernBERT improves AuthorMix only slightly because BERT
  already performs strongly on this dataset. The larger gain on PAN25
  suggests that the harder LLM-source attribution task benefits more from
  ModernBERT's longer-context and updated pretraining setup.

## 2. What tokens does the classifier use? (Content / Punctuation / Stop Words)

### Newline audit

ModernBERT's raw attribution output contains newline tokens on PAN25:

| Dataset | Method | Mean newline ratio | Median | Max |
| --- | --- | ---: | ---: | ---: |
| PAN25 | IG | 0.3329 | 0.3333 | 1.0000 |
| PAN25 | SHAP | 0.3154 | 0.2667 | 0.9333 |
| PAN25 | AR | 0.2102 | 0.2000 | 0.8000 |
| AuthorMix | IG | 0.0011 | 0.0000 | 0.2000 |
| AuthorMix | SHAP | 0.0010 | 0.0000 | 0.2000 |
| AuthorMix | AR | 0.0012 | 0.0000 | 0.2000 |

**Finding 1: PAN25 contains a strong ModernBERT newline signal.**\
This is not a graphing category for the thesis, but it is important for
clean comparison. If newline tokens were ignored or merged into punctuation,
the lexical ratios would be misleading. Therefore, the following content,
punctuation, and stopword ratios are non-newline normalised.

### PAN25 — IG non-newline-normalised ratios

| Class | IG Content Ratio | IG Punct Ratio | IG Stop Ratio |
| --- | ---: | ---: | ---: |
| **human** | 0.253 | 0.551 | 0.196 |
| gpt-4o | 0.219 | 0.634 | 0.147 |
| gpt-4-turbo-paraphrase | 0.352 | 0.501 | 0.148 |
| gpt-4.5-preview | **0.358** | 0.499 | 0.142 |
| gemini-pro-paraphrase | 0.346 | 0.529 | 0.125 |
| llama-3.3-70b-instruct | **0.143** | 0.633 | 0.225 |

**Finding 2: After newline correction, ModernBERT does not show a large
content-ratio shift on PAN25.**\
The previous interpretation that ModernBERT mainly shifted to content words
is no longer supported. The corrected IG content ratios mostly fall between
0.14 and 0.36. Punctuation remains the largest non-newline category for many
classes.

**Finding 3: PAN25 still shows class-specific lexical differences.**\
Some classes, such as gpt-4.5-preview and gpt-4-turbo-paraphrase, have
higher content ratios. Others, such as llama-3.3-70b-instruct, are strongly
punctuation/stopword-oriented. The signal is class-specific rather than a
simple human-versus-LLM split.

### AuthorMix — IG non-newline-normalised ratios

| Class | IG Content Ratio | IG Punct Ratio | IG Stop Ratio |
| --- | ---: | ---: | ---: |
| qq | **0.625** | 0.222 | 0.153 |
| pp | 0.555 | 0.318 | 0.127 |
| bush | 0.537 | 0.295 | 0.168 |
| blog5546 | 0.521 | 0.320 | 0.159 |
| woolf | 0.488 | 0.363 | 0.148 |
| hemingway | 0.447 | 0.349 | 0.204 |
| blog30102 | **0.411** | 0.368 | 0.221 |

**Finding 4: AuthorMix remains more content-oriented than PAN25.**\
Most AuthorMix classes have higher content ratios than PAN25 classes. This
supports the view that human author attribution uses more author-specific
and topic-specific vocabulary.

**Finding 5: ModernBERT's improvement is not explained by content ratio
alone.**\
The BERT-vs-ModernBERT non-newline IG content comparison is small:
PAN25 changes from 0.237 to 0.254, and AuthorMix from 0.464 to 0.475.
The performance gain therefore cannot be reduced to "more content words".
It likely reflects a combination of longer context, stronger pretraining,
tokenizer differences, and class-specific attribution patterns.

## 3. Attention Rollout after newline correction

### PAN25 — AR is less collapsed than BERT, but still punctuation-heavy

| Class | AR Content Ratio | AR Punct Ratio | AR Stop Ratio |
| --- | ---: | ---: | ---: |
| human | 0.133 | 0.597 | 0.270 |
| gpt-4o | 0.124 | 0.529 | 0.347 |
| gpt-4.5-preview | 0.176 | 0.535 | 0.289 |
| llama-3.1-8b-instruct | 0.193 | 0.485 | 0.322 |
| llama-3.3-70b-instruct | 0.161 | 0.416 | 0.423 |

**Finding 6: ModernBERT AR no longer collapses almost completely to
punctuation, but punctuation remains central.**\
Compared with BERT, ModernBERT AR has higher content and stopword ratios on
PAN25. Still, punctuation is usually the largest category. AR should
therefore be interpreted cautiously.

### AuthorMix — AR includes more content and author-specific punctuation

| Class | AR Content Ratio | AR Punct Ratio | AR Stop Ratio |
| --- | ---: | ---: | ---: |
| qq | 0.384 | 0.302 | 0.314 |
| trump | 0.365 | 0.351 | 0.283 |
| hemingway | 0.325 | 0.390 | 0.285 |
| obama | 0.326 | 0.391 | 0.284 |
| woolf | 0.253 | 0.516 | 0.231 |

**Finding 7: In AuthorMix, AR captures both content and punctuation style.**\
Woolf has a high AR punctuation ratio, which fits the expectation that
punctuation style matters for literary authors. Other classes show more
balanced AR profiles.

## 4. Cross-method agreement

### PAN25

| Method Pair | Mean Jaccard | Std |
| --- | ---: | ---: |
| IG-SHAP | **0.397** | 0.199 |
| IG-AR | 0.276 | 0.116 |
| SHAP-AR | 0.264 | 0.115 |

### AuthorMix

| Method Pair | Mean Jaccard | Std |
| --- | ---: | ---: |
| IG-SHAP | **0.617** | 0.216 |
| IG-AR | 0.404 | 0.193 |
| SHAP-AR | 0.405 | 0.191 |

**Finding 8: IG-SHAP remains the strongest pair, but ModernBERT changes the
agreement pattern.**\
PAN25 IG-SHAP agreement is lower under ModernBERT than under BERT, but
IG-AR and SHAP-AR are higher. This suggests that AR is less disconnected
from gradient-based methods in ModernBERT than in BERT.

**Finding 9: Agreement is stronger on AuthorMix than on PAN25.**\
The same pattern appears in the BERT report. AuthorMix gives more stable
top-token evidence across methods.

### Consensus tokens

- **PAN25:** consensus tokens still include many punctuation marks, but also
  occasional content and function words.
- **AuthorMix:** consensus tokens include author-specific content words,
  especially in political and literary classes.

**Finding 10:** ModernBERT consensus tokens are useful, but they should be
read as tokenizer-level evidence. Subword fragments and punctuation
combinations still appear in the fingerprint lists.

## 5. POS-tag analysis (stratified sample: 460 PAN25, 264 AuthorMix)

### PAN25 — IG POS distribution

- Nouns, prepositions, adverbs, and punctuation all appear in the top
  attributed tokens.
- The POS distribution is not dominated by one semantic category.
- The results fit the corrected ratio analysis: PAN25 remains mixed and
  partly structural after newline removal.

### AuthorMix — IG POS distribution

- Nouns are stronger in AuthorMix. For example, the output table shows noun
  proportions around 0.427 for bush, 0.376 for Woolf, and 0.363 for Obama
  in the sampled IG analysis.
- This supports the interpretation that AuthorMix is more lexical and
  topic-sensitive than PAN25.

**Finding 11: POS evidence supports the dataset contrast.**\
PAN25 uses a mixture of structural and lexical evidence. AuthorMix relies
more clearly on nouns and topic vocabulary.

## 6. Lexical sophistication

The lexical plots compare mean word length, subword ratio, stopword ratio,
and type-token ratio across methods and classes.

**Finding 12: Lexical sophistication is a secondary signal.**\
Mean word length and subword ratio vary across classes, but they do not
replace the main content/punctuation/stopword analysis.

**Finding 13: Type-token ratio is not the central explanation.**\
The classifier does not simply use vocabulary diversity in the top-15
tokens. Class differences are better captured by specific token types and
class-level vocabulary fingerprints.

## 7. Attribution concentration (entropy and Gini)

**Finding 14: Attribution is distributed rather than single-token based.**\
The concentration plots show that IG and SHAP usually spread attribution
across several tokens. This means the classifier is not using one obvious
word as a shortcut.

**Finding 15: AR remains more concentrated.**\
Attention Rollout often gives stronger weight to a smaller set of tokens,
especially punctuation and function-word-like tokens. This is consistent
with the method's attention-flow nature.

## 8. Positional analysis — where in the text does the classifier look?

### PAN25 — ModernBERT positions

- Under IG, the first-third share is often between 0.52 and 0.82.
- Human has first-third share 0.649.
- Several classes remain highly front-loaded, for example text-bison-002
  (0.822) and mistral-7b-instruct-v0.2 (0.809).
- Some classes are more distributed, for example gpt-4-turbo-paraphrase
  (0.524).

**Finding 16: ModernBERT still relies heavily on the beginning of PAN25
texts.**\
Even though ModernBERT supports longer context, the attribution signal on
PAN25 remains strongly front-loaded. The longer context does not remove
the opening-text bias.

### AuthorMix — ModernBERT positions

- AuthorMix is more evenly distributed than PAN25.
- IG first-third shares are often around 0.35--0.45.
- Classes such as qq and trump show more middle/end contribution than many
  PAN25 classes.

**Finding 17: AuthorMix uses broader document evidence.**\
This matches the BERT result and supports the view that human author
attribution is less dependent on the opening than PAN25 source attribution.

## 9. Statistical testing

### PAN25: human vs. GPT-4o

The following tests use non-newline-normalised lexical ratios.

| Method | Metric | Mean (human) | Mean (gpt-4o) | Cohen's $d$ | $p$-value |
| --- | --- | ---: | ---: | ---: | ---: |
| IG | Content | 0.253 | 0.219 | 0.207 | $4.41 \times 10^{-3}$ |
| IG | Punctuation | 0.551 | 0.634 | **-0.448** | $6.21 \times 10^{-9}$ |
| IG | Stopword | 0.196 | 0.147 | 0.353 | $2.76 \times 10^{-7}$ |
| SHAP | Content | 0.284 | 0.232 | 0.292 | $3.16 \times 10^{-5}$ |
| SHAP | Punctuation | 0.514 | 0.625 | **-0.570** | $5.89 \times 10^{-13}$ |
| SHAP | Stopword | 0.202 | 0.143 | 0.397 | $4.29 \times 10^{-9}$ |
| AR | Content | 0.133 | 0.124 | 0.090 | $2.06 \times 10^{-1}$ |
| AR | Punctuation | 0.597 | 0.529 | 0.385 | $2.09 \times 10^{-10}$ |
| AR | Stopword | 0.270 | 0.347 | **-0.550** | $2.92 \times 10^{-17}$ |

**Finding 18: Human and GPT-4o differ, but the channel is method-dependent.**\
The largest gradient-method contrast is punctuation: GPT-4o has a higher
punctuation ratio than human under IG and SHAP. AR shows the strongest
difference in stopwords.

### PAN25: human vs. GPT-4-turbo-paraphrase

| Method | Metric | Mean (human) | Mean (gpt-4-turbo-paraphrase) | Cohen's $d$ | $p$-value |
| --- | --- | ---: | ---: | ---: | ---: |
| IG | Content | 0.253 | 0.352 | **-0.606** | $2.08 \times 10^{-3}$ |
| IG | Punctuation | 0.551 | 0.501 | 0.271 | $1.10 \times 10^{-1}$ |
| IG | Stopword | 0.196 | 0.148 | 0.342 | $1.16 \times 10^{-2}$ |
| SHAP | Content | 0.284 | 0.361 | -0.418 | $1.51 \times 10^{-2}$ |
| AR | Content | 0.133 | 0.137 | -0.040 | $8.47 \times 10^{-1}$ |

**Finding 19: The human-vs-LLM difference depends on the LLM source.**\
GPT-4-turbo-paraphrase differs from human mainly in content ratio under IG
and SHAP, while AR barely separates them. This confirms that one global
"LLM feature" is not enough.

### AuthorMix examples

| Contrast | Main result |
| --- | --- |
| Hemingway vs. Woolf | AR punctuation has a large effect ($d=-0.999$), while AR content is higher for Hemingway ($d=0.651$). |
| Obama vs. Trump | IG and SHAP differences are small; AR content and punctuation show moderate effects. |

**Finding 20: AuthorMix differences are often class-pair specific.**\
The model distinguishes some author pairs through content, others through
punctuation style, and others only weakly through token-type proportions.

## 10. Vocabulary fingerprints

### PAN25 — exclusive tokens per class

- Human has the largest exclusive-token set in the output.
- Many LLM classes have smaller exclusive-token sets.
- Some exclusive tokens are subword fragments, numbers, and punctuation
  combinations. This is expected for token-level analysis and should not
  be read as clean word-level vocabulary.

**Finding 21: The fingerprint is useful, but it is a model-token
fingerprint.**\
For ModernBERT, byte-level tokenization means that some fingerprint entries
are fragments. The interpretation should focus on recurring patterns, not
isolated fragments.

### AuthorMix — topic and style signatures

- Obama and Trump show political vocabulary.
- Hemingway and Woolf show literary vocabulary and punctuation patterns.
- Blog authors contain informal and idiosyncratic tokens.

**Finding 22: AuthorMix fingerprints are more human-readable than PAN25
fingerprints.**\
A knowledgeable reader may recognize some AuthorMix signatures, but the
classifier still uses many token-level patterns below normal reading
awareness.

## 11. Answers to the questions

### Q1: What features are responsible for correct classification?

**For PAN25:**

1. Punctuation and stopword patterns among non-newline tokens.
2. A strong raw newline signal specific to ModernBERT on PAN25.
3. Class-specific content words for some LLM classes.
4. Strong opening-text dependence.

**For AuthorMix:**

1. Content words and nouns.
2. Author-specific topic vocabulary.
3. Punctuation style for some authors.
4. Broader use of the document than PAN25.

### Q2: Are LLM words different from human words?

Yes, but not as a simple content-word shift. After correcting for newline
tokens, the ModernBERT human-vs-LLM contrast depends on the LLM class and
the attribution method. GPT-4o differs from human mainly in punctuation and
stopword channels, while gpt-4-turbo-paraphrase differs more in content
ratio under IG and SHAP.

### Q3: Can a human reader identify the class just by reading?

Not reliably. Some AuthorMix classes have recognizable topics, but PAN25
differences include newline, punctuation, stopword, and positional patterns
that are not easy for a human reader to perceive. The classifier sees
distributional signals rather than obvious readable markers.

**Obfuscation implications:**

- For PAN25, modify opening structure, punctuation, and formatting.
- For ModernBERT specifically, check whether newline and paragraph layout
  are preserved, because they can become a classifier signal.
- For AuthorMix, content vocabulary substitution must be applied across the
  text, not only in the opening.
- A single obfuscation strategy is unlikely to transfer perfectly between
  BERT and ModernBERT.

## 12. Summary

| Dimension | PAN25 (LLM-source attribution) | AuthorMix (human author attribution) |
| --- | --- | --- |
| Balanced accuracy | $0.7967 \pm 0.0065$ | $0.8784 \pm 0.0041$ |
| BERT comparison | +18.1 pp | +2.1 pp |
| Raw newline audit | IG mean 0.3329; SHAP 0.3154; AR 0.2102 | approximately 0.001 |
| Thesis-facing ratios | non-newline-normalised | non-newline-normalised |
| IG content ratio | human 0.253; GPT-4o 0.219 | bush 0.537; qq 0.625 |
| AR behavior | punctuation-heavy but less collapsed than BERT | mixed content, stopword, and punctuation |
| IG-SHAP agreement | 0.397 | 0.617 |
| IG-AR agreement | 0.276 | 0.404 |
| Positional focus | still strongly front-loaded | more distributed |
| Main correction after newline audit | no large content-ratio shift | content-oriented author signal remains |
| Obfuscation target | opening structure + punctuation + formatting | full-text vocabulary and style |
| Human readability | difficult to detect by reading | partly detectable by topic |
