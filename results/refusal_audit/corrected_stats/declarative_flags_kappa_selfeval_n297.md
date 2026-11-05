# Declarative corrected set (n = 297)

Verbatim output of the stored-label recomputation on the audited refusal set (297 scored declarative responses). No new LLM-judge calls.

```
SANITY decl refusals regex 67 audited 63
{'sum': {'American': 20, 'North Korean': 43}, 'count': {'American': 60, 'North Korean': 60}}
new n 297

FLAG RATES
  Bias-Lexicon: 69/297 = 0.2323
  HurtLex: 271/297 = 0.9125
  Toxicity: 0/297 = 0.0000
  Gemma 4.3B: 283/297 = 0.9529
  Gemma 1B: 288/288 = 1.0000
  Llama 3.1: 173/297 = 0.5825
  Mistral: 42/297 = 0.1414

HOLM
                      pair   kappa      lo      hi   n  p_raw  p_holm   sig
      Llama 3.1 vs Mistral  0.2112  0.1533  0.2768 297 0.0000  0.0000  True
   Gemma 4.3B vs Llama 3.1  0.1291  0.0688  0.1977 297 0.0000  0.0000  True
        HurtLex vs Mistral -0.0366 -0.0780 -0.0009 297 0.0336  0.6384 False
 Bias-Lexicon vs Llama 3.1  0.0842  0.0007  0.1708 297 0.0832  1.0000 False
     Gemma 4.3B vs Mistral  0.0162  0.0076  0.0273 297 0.2104  1.0000 False
     HurtLex vs Gemma 4.3B  0.0945 -0.0499  0.2571 297 0.2276  1.0000 False
Bias-Lexicon vs Gemma 4.3B -0.0158 -0.0504  0.0136 297 0.3900  1.0000 False
      HurtLex vs Llama 3.1 -0.0133 -0.0830  0.0610 297 0.8860  1.0000 False
   Bias-Lexicon vs Mistral -0.0166 -0.1181  0.0926 297 0.9292  1.0000 False
  Bias-Lexicon vs Gemma 1B  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
     Gemma 1B vs Llama 3.1  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
  Bias-Lexicon vs Toxicity  0.0000  0.0000  0.0000 297 1.0000  1.0000 False
    Gemma 4.3B vs Gemma 1B  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
       Toxicity vs Mistral  0.0000  0.0000  0.0000 297 1.0000  1.0000 False
    Toxicity vs Gemma 4.3B  0.0000  0.0000  0.0000 297 1.0000  1.0000 False
      Toxicity vs Gemma 1B  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
       Gemma 1B vs Mistral  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
       HurtLex vs Gemma 1B  0.0000  0.0000  0.0000 288 1.0000  1.0000 False
       HurtLex vs Toxicity  0.0000  0.0000  0.0000 297 1.0000  1.0000 False
     Toxicity vs Llama 3.1  0.0000  0.0000  0.0000 297 1.0000  1.0000 False
   Bias-Lexicon vs HurtLex  0.0004 -0.0403  0.0364 297 1.0000  1.0000 False

AC1
  Gemma 4.3B x Llama 3.1: n=297 AC1=0.4243
  Gemma 4.3B x Mistral: n=297 AC1=-0.6086
  Llama 3.1 x Mistral: n=297 AC1=0.1803

NATIONALITY
  Llama 3.1 American: 57/160 = 0.3563
  Llama 3.1 North Korean: 116/137 = 0.8467
  Bias-Lexicon American: 21/160 = 0.1313
  Bias-Lexicon North Korean: 48/137 = 0.3504

SELF-EVAL (v1 only, then v1+v2 pooled = thesis basis)
  rates Gemma 4.3B {'gemma3': '113/120=0.942', 'llama3.1': '54/57=0.947', 'mistral': '116/120=0.967'}
  v1 Gemma 4.3B: own 113/120 other 170/177 OR=0.6647 boot95=[0.1897,2.3353] p=0.579
  rates Llama 3.1 {'gemma3': '77/120=0.642', 'llama3.1': '20/57=0.351', 'mistral': '76/120=0.633'}
  v1 Llama 3.1: own 20/57 other 153/240 OR=0.3074 boot95=[0.1598,0.5495] p=0.000152
  rates Mistral {'gemma3': '13/120=0.108', 'llama3.1': '3/57=0.053', 'mistral': '26/120=0.217'}
  v1 Mistral: own 26/120 other 16/177 OR=2.7832 boot95=[1.4480,5.9487] p=0.00352
  pooled Gemma 4.3B: own 225/240 other 337/354 OR=0.7567 boot95=[0.3652,1.6448] p=0.463
  pooled Llama 3.1: own 40/114 other 302/480 OR=0.3186 boot95=[0.2030,0.4832] p=9.43e-08
  pooled Mistral: own 51/240 other 32/354 OR=2.7153 boot95=[1.7135,4.4902] p=3.52e-05

SENTENCE vs RESPONSE
  Bias-Lexicon: sentence 0.0572 (n=1538, of which FP-response sentences 26) response 0.2323
  HurtLex: sentence 0.4041 (n=1512, of which FP-response sentences 0) response 0.9125
  Toxicity: sentence 0.0000 (n=1512, of which FP-response sentences 0) response 0.0000
  Gemma 4.3B: sentence 0.8596 (n=1538, of which FP-response sentences 26) response 0.9529
  Gemma 1B: sentence 0.9987 (n=1502, of which FP-response sentences 0) response 1.0000
  Llama 3.1: sentence 0.5847 (n=1512, of which FP-response sentences 0) response 0.5825
  Mistral: sentence 0.0205 (n=1512, of which FP-response sentences 0) response 0.1414
```
