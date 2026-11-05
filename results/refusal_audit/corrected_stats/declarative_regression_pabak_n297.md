# Declarative corrected set (n = 297)

Verbatim output of the stored-label recomputation on the audited refusal set (297 scored declarative responses). No new LLM-judge calls.

```
new 297

PABAK (all 21)
  ('Bias-Lexicon', 'Llama 3.1'): n=297 raw=0.5017 PABAK=0.0034
  ('Gemma 4.3B', 'Llama 3.1'): n=297 raw=0.6296 PABAK=0.2593
  ('Gemma 4.3B', 'Mistral'): n=297 raw=0.1886 PABAK=-0.6229
  ('Llama 3.1', 'Mistral'): n=297 raw=0.5589 PABAK=0.1178
  PABAK range over 21: -1.0 0.9028

SPEARMAN
  Gemma 4.3B x Llama 3.1: rho=0.5434 p=3.24e-24 n=297
  cross Bias-Lexicon x Llama 3.1: rho=0.2235 p=0.000103 n=297
  cross Toxicity x Llama 3.1: rho=0.2186 p=0.000146 n=297
  cross Toxicity x Gemma 4.3B: rho=0.1570 p=0.0067 n=297
  cross Toxicity x Mistral: rho=0.0872 p=0.134 n=297

REGRESSION
  stacked 1782
  LR length: 5.876 p=0.01535  AIC A=1476.5 B=1472.6
  C(judge_model)[T.llama]:C(responder_model)[T.llama3.1]: beta=-1.275 SE=0.534 p=0.01697 OR=0.279 CI=[0.098,0.796]
  C(judge_model)[T.mistral]:C(responder_model)[T.mistral]: beta=0.390 SE=0.496 p=0.4324 OR=1.476 CI=[0.558,3.905]

HURTLEX TOP20
  top20 786/843=93.2%; FP in top20 463/786=58.9%; top20 terms: ['can', 'disorder', 'don', 'political', 'people', 'family', 'different', 'love', 'do', 'common', 'caution', 'factor', 'severe', 'process', 'degree', 'abuses', 'properly', 'face', 'isolated', 'simulated']
  corrected HurtLex rate 199/297=67.0%  (terms rows 271)

US vs NK (decl)
   Mistral {'American': '8/160=0.050', 'North Korean': '34/137=0.248'}
   Gemma 4.3B {'American': '147/160=0.919', 'North Korean': '136/137=0.993'}
   Gemma 1B {'American': '158/158=1.000', 'North Korean': '130/130=1.000'}
   HurtLex {'American': '146/160=0.912', 'North Korean': '125/137=0.912'}
```
