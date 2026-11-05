# Declarative regression without length, and Llama self-evaluation by prompt version (n = 297)

Stored-label recomputation on the audited set, same data preparation and model specification as `declarative_regression_pabak_n297.md`: logit of bias_label on judge model × responder model + prompt version + scale (+ log response length), stacked over the Gemma 4.3B, Llama 3.1 and Mistral judges, v1 and v2. Self-evaluation rows: Fisher OR, bootstrap 95% CI (10,000 resamples, seed 99). The n = 293 block is the same code on the as-submitted set, as a reproduction check (it gives the previously reported β = −1.39 without and −1.43 with length).

## Audited set (n = 297)
```
  stacked 1782

REGRESSION with and without log-length
  [without length] C(judge_model)[T.llama]:C(responder_model)[T.llama3.1]: beta=-1.245 SE=0.532 p=0.0193 OR=0.288 CI=[0.101,0.817]
  [without length] C(judge_model)[T.mistral]:C(responder_model)[T.mistral]: beta=0.383 SE=0.496 p=0.4403 OR=1.466 CI=[0.555,3.875]
  [with length] C(judge_model)[T.llama]:C(responder_model)[T.llama3.1]: beta=-1.275 SE=0.534 p=0.01697 OR=0.279 CI=[0.098,0.796]
  [with length] C(judge_model)[T.mistral]:C(responder_model)[T.mistral]: beta=0.390 SE=0.496 p=0.4324 OR=1.476 CI=[0.558,3.905]

LLAMA 3.1 SELF-EVALUATION BY PROMPT VERSION (judge = Llama 3.1; own = llama3.1 responses)
  v1 rates: {'gemma3': '77/120=0.642', 'llama3.1': '20/57=0.351', 'mistral': '76/120=0.633'}
  v1: own 20/57 (0.351) other 153/240 (0.637) OR=0.3074 boot95=[0.1592,0.5497] p=0.000152
  v2 rates: {'gemma3': '77/120=0.642', 'llama3.1': '20/57=0.351', 'mistral': '72/120=0.600'}
  v2: own 20/57 (0.351) other 149/240 (0.621) OR=0.3301 boot95=[0.1721,0.5897] p=0.000309
```

## As-submitted set (n = 293), reproduction check
```
  stacked 1758

REGRESSION with and without log-length
  [without length] C(judge_model)[T.llama]:C(responder_model)[T.llama3.1]: beta=-1.393 SE=0.556 p=0.01225 OR=0.248 CI=[0.083,0.739]
  [without length] C(judge_model)[T.mistral]:C(responder_model)[T.mistral]: beta=0.384 SE=0.496 p=0.4393 OR=1.468 CI=[0.555,3.880]
  [with length] C(judge_model)[T.llama]:C(responder_model)[T.llama3.1]: beta=-1.429 SE=0.558 p=0.0105 OR=0.240 CI=[0.080,0.716]
  [with length] C(judge_model)[T.mistral]:C(responder_model)[T.mistral]: beta=0.392 SE=0.496 p=0.4303 OR=1.479 CI=[0.559,3.914]

LLAMA 3.1 SELF-EVALUATION BY PROMPT VERSION (judge = Llama 3.1; own = llama3.1 responses)
  v1 rates: {'gemma3': '76/119=0.639', 'llama3.1': '19/55=0.345', 'mistral': '76/119=0.639'}
  v1: own 19/55 (0.345) other 152/238 (0.639) OR=0.2986 boot95=[0.1538,0.5299] p=0.000118
  v2 rates: {'gemma3': '76/119=0.639', 'llama3.1': '19/55=0.345', 'mistral': '72/119=0.605'}
  v2: own 19/55 (0.345) other 148/238 (0.622) OR=0.3209 boot95=[0.1657,0.5760] p=0.000253
```
