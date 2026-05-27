# Feature Attribution Analysis Report

## What Features Drive BERT's Authorship Classification?

**Thesis goal:** Identify which text features are responsible for correct
classification by the fine-tuned BERT authorship classifier, and use these
findings to define possible future obfuscation targets. The report asks whether
LLM-generated texts differ from human-written texts at the feature level, whether
a reader could identify the source class by reading, and which token types the
classifier uses.\
**Data:** Full validation sets: PAN25 (3,556 examples, 23 classes) and
AuthorMix (3,642 examples, 14 classes).\
**Methods:** Integrated Gradients (IG), GradientSHAP (SHAP), and Attention
Rollout (AR), all applied to every validation example.\
**Model:** BERT (`bert-base-uncased`) fine-tuned classifiers. Mean balanced
accuracy across three seeds: PAN25 $0.6152 \pm 0.0088$ and AuthorMix
$0.8572 \pm 0.0052$.\
**Token-ratio update:** The ratio computation now uses four raw categories:
content, punctuation, stopword, and newline. For BERT, the newline ratio is
zero for PAN25 and AuthorMix, so the raw lexical ratios and the non-newline
lexical ratios are identical.

## 1. Classification performance context

### PAN25 Multiclass (23 LLM sources + human)

- **Mean balanced accuracy across three seeds:** $0.6152 \pm 0.0088$.
  This is a difficult 23-way classification problem.

- **Easy to classify:** human text is one of the easier classes for BERT
  (reported per-class accuracy: 91.9%). Several LLM classes are also
  classified well, but the model struggles to separate closely related
  LLM sources.

- **Hard to classify:** mixtral-8x7b-instruct-v0.1, llama-2-70b-chat,
  gpt-4o, and llama-3.1-8b-instruct are among the weaker classes in the
  BERT baseline.

- **Key point:** Human text is easier to separate from many LLM classes
  than the LLM classes are from each other. This suggests that the
  23-way PAN25 task is not only a human-versus-machine task, but also a
  fine-grained source attribution task among similar generation systems.

### AuthorMix (14 human authors)

- **Mean balanced accuracy across three seeds:** $0.8572 \pm 0.0052$.

- **Strong classes:** the small classes `h`, `pp`, and `qq` are classified
  particularly well in the baseline outputs.

- **Harder classes:** `bush`, `blog30102`, and `trump` are less cleanly
  separated than the smallest distinctive classes.

- **Key point:** AuthorMix is easier than PAN25 for BERT. The result
  suggests that individual human authors in AuthorMix provide stronger
  closed-set fingerprints than the 23 PAN25 source classes.

## 2. What tokens does the classifier use? (Content / Punctuation / Stop Words)

### PAN25 — IG ratios (top-15 attributed tokens per example)

The following values are raw BERT ratios. Because BERT has newline ratio
$0.000$ in this analysis, these values are also the non-newline ratios.

| Class | IG Content Ratio | IG Punct Ratio | IG Stop Ratio |
| --- | ---: | ---: | ---: |
| **human** | **0.215** | 0.334 | **0.451** |
| gpt-4o | 0.288 | 0.295 | 0.416 |
| gpt-4.5-preview | 0.363 | 0.249 | 0.389 |
| mixtral-8x7b-instruct-v0.1 | 0.374 | 0.321 | 0.305 |
| llama-3.1-8b-instruct | **0.165** | 0.362 | **0.473** |
| gemini-1.5-pro | 0.190 | 0.323 | 0.486 |

**Finding 1: BERT uses function-word and punctuation patterns strongly on
PAN25.**\
The human class has low content attribution (0.215) and high stopword
attribution (0.451). Some LLM classes have even stronger stopword patterns,
for example gemini-1.5-pro (0.486) and llama-3.1-8b-instruct (0.473).
This indicates that BERT is not only using topic vocabulary. It relies on
how language is structured: function words, connectives, and punctuation.

**Finding 2: Some LLM classes look close to human text in token-type
profile.**\
The attribution profile of llama-3.1-8b-instruct is close to the human
profile: low content ratio and high stopword ratio. This fits the weaker
classification of this class in the BERT baseline. The classifier appears
to struggle when an LLM source shares function-word and punctuation patterns
with human writing.

**Finding 3: Higher content ratios are class-specific, not a general LLM
property.**\
Some classes, such as gpt-4.5-preview and mixtral-8x7b-instruct-v0.1,
have higher content ratios. This does not mean that all LLMs are detected
through content words. It means that for some source classes BERT uses
more lexical or topical cues, while for others it relies more on
function-word patterns.

### AuthorMix — IG ratios

| Class | IG Content Ratio | IG Punct Ratio | IG Stop Ratio |
| --- | ---: | ---: | ---: |
| bush | **0.593** | 0.134 | 0.273 |
| woolf | 0.548 | 0.191 | 0.261 |
| blog5546 | 0.544 | 0.210 | 0.246 |
| blog11518 | 0.381 | 0.246 | 0.374 |
| blog30102 | 0.380 | 0.224 | 0.396 |

**Finding 4: AuthorMix relies more on content vocabulary than PAN25.**\
In AuthorMix, several author classes have high content ratios. Bush has
the highest IG content ratio among the listed classes (0.593), and Woolf
and blog5546 are also above 0.54. This suggests that BERT often identifies
human authors through their characteristic vocabulary and domain-specific
word choices.

**Finding 5: PAN25 and AuthorMix use different feature mixtures.**

- PAN25: the human-vs-LLM setting is strongly shaped by stopwords and
  punctuation, especially under BERT.
- AuthorMix: author identity is more strongly associated with content
  vocabulary and topic-specific nouns.

This is one of the central findings. The same encoder family does not use
the same feature mixture on both authorship tasks.

## 3. Attention Rollout reveals punctuation dominance

### PAN25 — AR collapses to punctuation

- AR content ratio across PAN25 classes is very low, roughly $0.003$ to
  $0.037$.
- AR punctuation ratio is very high, roughly $0.95$ to $0.99$.
- Human has AR content ratio $0.030$ and AR punctuation ratio $0.960$.
- GPT-4o has AR content ratio $0.004$ and AR punctuation ratio $0.992$.

**Finding 6: Attention Rollout is dominated by punctuation in BERT PAN25.**\
This does not mean punctuation alone is the semantic reason for the
prediction. It suggests that BERT attention uses punctuation and delimiters
as aggregation points. AR therefore behaves more like an attention-flow
diagnostic than a faithful semantic explanation on this task.

### AuthorMix — AR is less collapsed

- AR content ratios are much higher than on PAN25, for example bush 0.299,
  hemingway 0.337, and blog30407 0.394.
- AR punctuation ratios remain substantial, for example Woolf 0.586 and
  pp 0.502.

**Finding 7: AR is more informative on AuthorMix than on PAN25.**\
In AuthorMix, AR still gives large weight to punctuation, but it no longer
collapses almost completely to punctuation. It captures some content and
author-specific surface style.

## 4. Cross-method agreement

### PAN25

| Method Pair | Mean Jaccard | Std |
| --- | ---: | ---: |
| IG-SHAP | **0.547** | 0.172 |
| IG-AR | 0.147 | 0.086 |
| SHAP-AR | 0.144 | 0.087 |

### AuthorMix

| Method Pair | Mean Jaccard | Std |
| --- | ---: | ---: |
| IG-SHAP | **0.704** | 0.210 |
| IG-AR | 0.360 | 0.196 |
| SHAP-AR | 0.358 | 0.195 |

**Finding 8: IG and SHAP agree strongly, while AR diverges.**\
The two gradient-based methods have the strongest agreement in both
datasets. This supports the reliability of the gradient-based attribution
patterns. AR has much lower overlap on PAN25 because its top tokens are
mostly punctuation.

**Finding 9: Agreement depends on both method and task.**\
AuthorMix has higher agreement than PAN25 for all method pairs. This
suggests that author attribution yields more stable token-level evidence
than the 23-way PAN25 source attribution setting.

### Consensus tokens

- **PAN25:** consensus tokens are mostly punctuation marks and common
  function words.
- **AuthorMix:** consensus tokens include more recognizable content words,
  especially for political and literary authors.

**Finding 10:** When all three methods agree, the token is strong evidence
for the classifier's decision. In PAN25 this agreement often falls on
structural tokens; in AuthorMix it more often includes lexical signatures.

## 5. POS-tag analysis (stratified sample: 460 PAN25, 264 AuthorMix)

### PAN25 — IG POS distribution

- Punctuation and function-word-related POS categories are frequent among
  the top attributed tokens.
- Human text has a high preposition/function-word component.
- Noun ratios vary by source class, but nouns are not the sole dominant
  signal in PAN25.

### AuthorMix — IG POS distribution

- Nouns are stronger in AuthorMix than in PAN25. In the sample table,
  bush has a noun proportion around 0.403.
- Punctuation remains present but is less dominant than in PAN25.
- Verb ratios are more uniform and are less useful as a class-level
  discriminator.

**Finding 11: BERT identifies authors partly through noun choice and
domain vocabulary.**\
For AuthorMix, the POS results support the token-ratio findings. Author
classification uses more content-bearing words than PAN25.

## 6. Lexical sophistication

The lexical plots compare mean word length, subword ratio, stopword ratio,
and type-token ratio across classes and attribution methods.

**Finding 12: Lexical metrics provide secondary evidence rather than the
main signal.**\
Mean word length and subword ratio vary across classes, but they are not
as directly interpretable as the content, punctuation, and stopword ratios.

**Finding 13: Type-token ratio is not the main discriminator.**\
The type-token ratio of the top attributed tokens is relatively stable
across many classes. BERT does not simply distinguish classes by vocabulary
diversity in the top-15 tokens.

## 7. Attribution concentration (entropy and Gini)

**Finding 14: IG and SHAP distribute attribution across several tokens.**\
The high normalized entropy values indicate that the classifier does not
usually rely on a single token. It combines several weak or medium-strength
signals.

**Finding 15: AR is more concentrated than the gradient methods.**\
This is especially visible on PAN25, where a small number of punctuation
tokens can dominate the attention-rollout score.

## 8. Positional analysis — where in the text does the classifier look?

### PAN25 — IG positions

- Many PAN25 classes are strongly front-loaded under BERT.
- The first third receives the majority of top attributions for most
  classes.
- This is consistent with BERT's 512-token limit, because only the early
  part of long documents is available.

**Finding 16: BERT relies strongly on the beginning of PAN25 texts.**\
This matters for obfuscation. If the opening of the text is rewritten,
the classifier may lose a large part of its evidence.

### AuthorMix — IG positions

- AuthorMix is less front-loaded than PAN25.
- Attribution is more spread across the beginning, middle, and end of the
  available text.

**Finding 17: Author attribution uses broader textual evidence.**\
The author signal is less confined to the opening than the PAN25 LLM-source
signal.

## 9. Statistical testing

### PAN25: human vs. GPT-4o

| Method | Metric | Mean (human) | Mean (gpt-4o) | Cohen's $d$ | $p$-value |
| --- | --- | ---: | ---: | ---: | ---: |
| IG | Content Ratio | 0.215 | 0.288 | **-0.561** | $2.83 \times 10^{-13}$ |
| IG | Punct Ratio | 0.334 | 0.295 | 0.295 | $5.15 \times 10^{-5}$ |
| IG | Stop Ratio | 0.451 | 0.416 | 0.230 | $1.95 \times 10^{-3}$ |
| SHAP | Content Ratio | 0.230 | 0.288 | **-0.407** | $2.05 \times 10^{-7}$ |
| AR | Content Ratio | 0.030 | 0.004 | **0.538** | $1.61 \times 10^{-45}$ |
| AR | Punct Ratio | 0.960 | 0.992 | **-0.519** | $2.98 \times 10^{-31}$ |

**Finding 18: Human and GPT-4o differ significantly in token-type profile.**\
Human text has lower content ratio and higher stopword ratio under IG.
Under AR, the contrast is dominated by punctuation.

**Finding 19: GPT-4o is identified more through lexical content than human
text under BERT.**\
The effect is not large enough to reduce the task to a simple rule, but it
is consistent across the gradient methods.

### AuthorMix: Bush vs. PP

| Method | Metric | Mean (bush) | Mean (pp) | Cohen's $d$ | $p$-value |
| --- | --- | ---: | ---: | ---: | ---: |
| IG | Content Ratio | 0.593 | 0.518 | 0.499 | $5.51 \times 10^{-2}$ |
| IG | Punct Ratio | 0.134 | 0.157 | -0.265 | $2.40 \times 10^{-1}$ |
| IG | Stop Ratio | 0.273 | 0.325 | -0.434 | $4.21 \times 10^{-2}$ |
| SHAP | Content Ratio | 0.583 | 0.545 | 0.252 | $2.72 \times 10^{-1}$ |
| AR | Content Ratio | 0.299 | 0.278 | 0.180 | $5.04 \times 10^{-1}$ |

**Finding 20: Bush vs. PP is not explained cleanly by token-type proportions.**\
The IG stopword ratio reaches conventional significance in the Welch test,
but the overall pattern is weak and not stable across methods. This means
the classifier likely separates these authors through *which* words occur,
not only through the proportion of content, stopword, and punctuation
tokens.

## 10. Vocabulary fingerprints

### PAN25 — exclusive tokens per class

- Human text has the largest set of exclusive top-attributed tokens.
- LLM classes have smaller and more constrained exclusive vocabularies.
- Many exclusive tokens are subword fragments or punctuation combinations,
  so the list should be interpreted as a model-token fingerprint, not as a
  clean human-readable vocabulary list.

**Finding 21: Human text has the richest attributed-token fingerprint.**\
This likely reflects the broad topical and stylistic range of the human
class.

### AuthorMix — consensus tokens reveal topic signatures

- Obama examples include civic and policy vocabulary such as "Americans",
  "innovation", and "immigration".
- Literary authors show more author-specific lexical and punctuation
  patterns.
- Blog authors show informal and idiosyncratic surface tokens.

**Finding 22: Vocabulary fingerprints are useful but topic-sensitive.**\
They can explain why the classifier separates classes, but they do not
prove that the model has learned topic-independent authorship style.

## 11. Answers to the questions

### Q1: What features are responsible for correct classification?

**For PAN25:**

1. Stopword and function-word distributions.
2. Punctuation patterns.
3. Strong opening-text dependence.
4. Distributed evidence across several tokens, especially for IG and SHAP.

**For AuthorMix:**

1. Content words and nouns.
2. Author-specific vocabulary.
3. Broader use of the text, not only the opening.
4. Some punctuation style, especially for literary authors.

### Q2: Are LLM words different from human words?

Yes, but the difference is subtle. BERT does not simply detect "AI words".
Human and LLM classes differ in the distribution of function words,
punctuation, and selected content words. GPT-4o has a higher IG content
ratio than human text, while human text has a higher stopword ratio.

### Q3: Can a human reader identify the class just by reading?

Not reliably. The strongest BERT signals are token-distribution patterns,
not obvious readable cues. A reader may recognize some author-specific
topics in AuthorMix, but would not reliably perceive the function-word and
punctuation patterns used in PAN25.

**Obfuscation implications:**

- For PAN25, rewrite the opening and alter function-word/punctuation
  patterns.
- For AuthorMix, change topic-specific vocabulary across the whole text.
- Because the attribution evidence is distributed, one-word substitution is
  unlikely to be sufficient.

## 12. Summary

| Dimension | PAN25 (LLM-source attribution) | AuthorMix (human author attribution) |
| --- | --- | --- |
| Balanced accuracy | $0.6152 \pm 0.0088$ | $0.8572 \pm 0.0052$ |
| Newline audit | 0.000 for all methods | 0.000 for all methods |
| Primary signal | Stopwords + punctuation | Content words + topic vocabulary |
| IG content ratio | human 0.215; GPT-4o 0.288 | bush 0.593; woolf 0.548 |
| AR behavior | 95--99% punctuation | mixed content and punctuation |
| IG-SHAP agreement | 0.547 | 0.704 |
| IG-AR agreement | 0.147 | 0.360 |
| Positional focus | strongly front-loaded | more distributed |
| Example statistical contrast | human vs. GPT-4o: IG content $d=-0.561$ | bush vs. PP: weak / method-dependent |
| Obfuscation target | opening text + function words + punctuation | full-text topic vocabulary |
| Human readability | difficult to detect by reading | partly detectable for known authors |
