# Árvore de decisão do HMDA 2024

Esta árvore documenta a divisão anterior, com 1.746.384 pedidos para ajuste. O protocolo atual usa 70% para treino, 20% para validação e 10% para teste. A árvore continua disponível para explicar como seguir os ramos.

Esta é a primeira árvore da floresta aleatória na configuração A+B+C. Ela foi reproduzida separadamente com o mesmo pré-processamento, os mesmos pedidos de ajuste e a mesma semente do notebook. A floresta completa combina as probabilidades de 100 árvores e aplica o limiar definido na validação.

A base validada tem 2.910.642 pedidos, dos quais 1.746.384 foram reservados ao ajuste. A árvore tem 2.681 nós, 1.341 folhas e profundidade 20. Usamos semente 42 e scikit-learn 1.9.1. [Notebook de treinamento](../notebooks/02_treinamento_recorte_integral.ipynb)

Comece no nó 0, responda à pergunta e siga o ramo Sim ou Não até uma folha. A previsão da folha usa a classe mais frequente nela. As contagens mostram as observações recebidas pelo nó na amostra com reposição da árvore, então um pedido pode aparecer mais de uma vez.

## Árvore completa

```text
NÓ 0 | applicant_age = '45-54'? | bootstrap: 204765 recusados, 1541619 aprovados
├─ Sim: NÓ 2128 | property_value (valor do imóvel em USD) <= 130000.0? | bootstrap: 36948 recusados, 238971 aprovados
│   ├─ Sim: NÓ 2129 | lei = '5493002B2WMHY23GFK92'? | bootstrap: 7920 recusados, 9559 aprovados
│   │   ├─ Sim: FOLHA 2169 | previsão: APROVADO (1) | bootstrap: 7 recusados, 23 aprovados
│   │   └─ Não: NÓ 2130 | lei = '549300Q3ROEQU3D1IW91'? | bootstrap: 7913 recusados, 9536 aprovados
│   │       ├─ Sim: FOLHA 2168 | previsão: RECUSADO (0) | bootstrap: 550 recusados, 188 aprovados
│   │       └─ Não: NÓ 2131 | lei = '5493005RDTT7HVODH436'? | bootstrap: 7363 recusados, 9348 aprovados
│   │           ├─ Sim: FOLHA 2167 | previsão: APROVADO (1) | bootstrap: 6 recusados, 34 aprovados
│   │           └─ Não: NÓ 2132 | lei = '549300R0SXC1JU7ZU290'? | bootstrap: 7357 recusados, 9314 aprovados
│   │               ├─ Sim: FOLHA 2166 | previsão: APROVADO (1) | bootstrap: 8 recusados, 39 aprovados
│   │               └─ Não: NÓ 2133 | lei = 'QFROUN1UWUYU0DVIWD51'? | bootstrap: 7349 recusados, 9275 aprovados
│   │                   ├─ Sim: NÓ 2163 | loan_type = '2'? | bootstrap: 17 recusados, 50 aprovados
│   │                   │   ├─ Sim: FOLHA 2165 | previsão: APROVADO (1) | bootstrap: 4 recusados, 10 aprovados
│   │                   │   └─ Não: FOLHA 2164 | previsão: APROVADO (1) | bootstrap: 13 recusados, 40 aprovados
│   │                   └─ Não: NÓ 2134 | lei = 'S0Q3AHZRL5K6VQE35M07'? | bootstrap: 7332 recusados, 9225 aprovados
│   │                       ├─ Sim: FOLHA 2162 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
│   │                       └─ Não: NÓ 2135 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 56.5? | bootstrap: 7332 recusados, 9208 aprovados
│   │                           ├─ Sim: NÓ 2136 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 607 recusados, 1144 aprovados
│   │                           │   ├─ Sim: FOLHA 2142 | previsão: APROVADO (1) | bootstrap: 176 recusados, 251 aprovados
│   │                           │   └─ Não: NÓ 2137 | debt_to_income_ratio = '47'? | bootstrap: 431 recusados, 893 aprovados
│   │                           │       ├─ Sim: FOLHA 2141 | previsão: APROVADO (1) | bootstrap: 6 recusados, 29 aprovados
│   │                           │       └─ Não: NÓ 2138 | derived_race = 'Joint'? | bootstrap: 425 recusados, 864 aprovados
│   │                           │           ├─ Sim: FOLHA 2140 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 6 aprovados
│   │                           │           └─ Não: FOLHA 2139 | previsão: APROVADO (1) | bootstrap: 418 recusados, 858 aprovados
│   │                           └─ Não: NÓ 2143 | construction_method = '1'? | bootstrap: 6725 recusados, 8064 aprovados
│   │                               ├─ Sim: NÓ 2147 | loan_term (prazo em meses) <= 318.0? | bootstrap: 1190 recusados, 4850 aprovados
│   │                               │   ├─ Sim: NÓ 2148 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 216 recusados, 675 aprovados
│   │                               │   │   ├─ Sim: NÓ 2152 | derived_race = 'White'? | bootstrap: 30 recusados, 180 aprovados
│   │                               │   │   │   ├─ Sim: FOLHA 2156 | previsão: APROVADO (1) | bootstrap: 26 recusados, 151 aprovados
│   │                               │   │   │   └─ Não: NÓ 2153 | lien_status = '1'? | bootstrap: 4 recusados, 29 aprovados
│   │                               │   │   │       ├─ Sim: FOLHA 2155 | previsão: APROVADO (1) | bootstrap: 3 recusados, 22 aprovados
│   │                               │   │   │       └─ Não: FOLHA 2154 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
│   │                               │   │   └─ Não: NÓ 2149 | derived_race = 'White'? | bootstrap: 186 recusados, 495 aprovados
│   │                               │   │       ├─ Sim: FOLHA 2151 | previsão: APROVADO (1) | bootstrap: 108 recusados, 403 aprovados
│   │                               │   │       └─ Não: FOLHA 2150 | previsão: APROVADO (1) | bootstrap: 78 recusados, 92 aprovados
│   │                               │   └─ Não: NÓ 2157 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 66.5? | bootstrap: 974 recusados, 4175 aprovados
│   │                               │       ├─ Sim: FOLHA 2158 | previsão: APROVADO (1) | bootstrap: 179 recusados, 521 aprovados
│   │                               │       └─ Não: NÓ 2159 | lei = '54930054ET8KM4O1D485'? | bootstrap: 795 recusados, 3654 aprovados
│   │                               │           ├─ Sim: FOLHA 2161 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 2 aprovados
│   │                               │           └─ Não: FOLHA 2160 | previsão: APROVADO (1) | bootstrap: 790 recusados, 3652 aprovados
│   │                               └─ Não: NÓ 2144 | debt_to_income_ratio = '40'? | bootstrap: 5535 recusados, 3214 aprovados
│   │                                   ├─ Sim: FOLHA 2146 | previsão: APROVADO (1) | bootstrap: 111 recusados, 128 aprovados
│   │                                   └─ Não: FOLHA 2145 | previsão: RECUSADO (0) | bootstrap: 5424 recusados, 3086 aprovados
│   └─ Não: NÓ 2170 | lei = '549300OBO7DOF2KOP535'? | bootstrap: 29028 recusados, 229412 aprovados
│       ├─ Sim: NÓ 2670 | debt_to_income_ratio = '47'? | bootstrap: 198 recusados, 281 aprovados
│       │   ├─ Sim: NÓ 2678 | income (renda anual declarada em milhares de USD) <= 95.0? | bootstrap: 6 recusados, 14 aprovados
│       │   │   ├─ Sim: FOLHA 2679 | previsão: APROVADO (1) | bootstrap: 3 recusados, 11 aprovados
│       │   │   └─ Não: FOLHA 2680 | previsão: RECUSADO (0) | bootstrap: 3 recusados, 3 aprovados
│       │   └─ Não: NÓ 2671 | debt_to_income_ratio = '46'? | bootstrap: 192 recusados, 267 aprovados
│       │       ├─ Sim: NÓ 2675 | loan_term (prazo em meses) <= 330.0? | bootstrap: 7 recusados, 5 aprovados
│       │       │   ├─ Sim: FOLHA 2676 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 1 aprovados
│       │       │   └─ Não: FOLHA 2677 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
│       │       └─ Não: NÓ 2672 | debt_to_income_ratio = '39'? | bootstrap: 185 recusados, 262 aprovados
│       │           ├─ Sim: FOLHA 2674 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
│       │           └─ Não: FOLHA 2673 | previsão: APROVADO (1) | bootstrap: 184 recusados, 248 aprovados
│       └─ Não: NÓ 2171 | loan_term (prazo em meses) <= 351.5? | bootstrap: 28830 recusados, 229131 aprovados
│           ├─ Sim: NÓ 2172 | lei = '549300MNZECUR067HB12'? | bootstrap: 5389 recusados, 17372 aprovados
│           │   ├─ Sim: FOLHA 2358 | previsão: APROVADO (1) | bootstrap: 0 recusados, 37 aprovados
│           │   └─ Não: NÓ 2173 | lien_status = '2'? | bootstrap: 5389 recusados, 17335 aprovados
│           │       ├─ Sim: NÓ 2285 | lei = '549300DMYENRV818D594'? | bootstrap: 1207 recusados, 5549 aprovados
│           │       │   ├─ Sim: FOLHA 2357 | previsão: APROVADO (1) | bootstrap: 2 recusados, 14 aprovados
│           │       │   └─ Não: NÓ 2286 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 123.5? | bootstrap: 1205 recusados, 5535 aprovados
│           │       │       ├─ Sim: NÓ 2287 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 782 recusados, 4079 aprovados
│           │       │       │   ├─ Sim: FOLHA 2331 | previsão: APROVADO (1) | bootstrap: 18 recusados, 55 aprovados
│           │       │       │   └─ Não: NÓ 2288 | lei = '549300B4IYL7TZT8FA34'? | bootstrap: 764 recusados, 4024 aprovados
│           │       │       │       ├─ Sim: FOLHA 2330 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
│           │       │       │       └─ Não: NÓ 2289 | lei = 'DRMSV1Q0EKMEXLAU1P80'? | bootstrap: 764 recusados, 4010 aprovados
│           │       │       │           ├─ Sim: FOLHA 2329 | previsão: APROVADO (1) | bootstrap: 64 recusados, 105 aprovados
│           │       │       │           └─ Não: NÓ 2290 | loan_term (prazo em meses) <= 183.0? | bootstrap: 700 recusados, 3905 aprovados
│           │       │       │               ├─ Sim: NÓ 2291 | lei = '549300KJ8PAJ7E52HG32'? | bootstrap: 465 recusados, 3184 aprovados
│           │       │       │               │   ├─ Sim: FOLHA 2313 | previsão: APROVADO (1) | bootstrap: 0 recusados, 24 aprovados
│           │       │       │               │   └─ Não: NÓ 2292 | lei = '549300HYHFCULSUBAE54'? | bootstrap: 465 recusados, 3160 aprovados
│           │       │       │               │       ├─ Sim: FOLHA 2312 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
│           │       │       │               │       └─ Não: NÓ 2293 | lei = '01J4SO3XTWZF4PP38209'? | bootstrap: 465 recusados, 3135 aprovados
│           │       │       │               │           ├─ Sim: NÓ 2309 | income (renda anual declarada em milhares de USD) <= 67.0? | bootstrap: 1 recusados, 19 aprovados
│           │       │       │               │           │   ├─ Sim: FOLHA 2310 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
│           │       │       │               │           │   └─ Não: FOLHA 2311 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
│           │       │       │               │           └─ Não: NÓ 2294 | debt_to_income_ratio = '40'? | bootstrap: 464 recusados, 3116 aprovados
│           │       │       │               │               ├─ Sim: FOLHA 2308 | previsão: APROVADO (1) | bootstrap: 14 recusados, 145 aprovados
│           │       │       │               │               └─ Não: NÓ 2295 | lei = '8I3UVGYULPJQIP7FQV10'? | bootstrap: 450 recusados, 2971 aprovados
│           │       │       │               │                   ├─ Sim: FOLHA 2307 | previsão: APROVADO (1) | bootstrap: 23 recusados, 45 aprovados
│           │       │       │               │                   └─ Não: NÓ 2296 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 427 recusados, 2926 aprovados
│           │       │       │               │                       ├─ Sim: NÓ 2304 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 42.0? | bootstrap: 20 recusados, 372 aprovados
│           │       │       │               │                       │   ├─ Sim: FOLHA 2305 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
│           │       │       │               │                       │   └─ Não: FOLHA 2306 | previsão: APROVADO (1) | bootstrap: 17 recusados, 365 aprovados
│           │       │       │               │                       └─ Não: NÓ 2297 | lei = 'X05BVSK68TQ7YTOSNR22'? | bootstrap: 407 recusados, 2554 aprovados
│           │       │       │               │                           ├─ Sim: NÓ 2301 | income (renda anual declarada em milhares de USD) <= 104.5? | bootstrap: 14 recusados, 3 aprovados
│           │       │       │               │                           │   ├─ Sim: FOLHA 2302 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 1 aprovados
│           │       │       │               │                           │   └─ Não: FOLHA 2303 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 2 aprovados
│           │       │       │               │                           └─ Não: NÓ 2298 | lei = 'IUGPUX5LWRZ3B6KIZ697'? | bootstrap: 393 recusados, 2551 aprovados
│           │       │       │               │                               ├─ Sim: FOLHA 2300 | previsão: APROVADO (1) | bootstrap: 6 recusados, 63 aprovados
│           │       │       │               │                               └─ Não: FOLHA 2299 | previsão: APROVADO (1) | bootstrap: 387 recusados, 2488 aprovados
│           │       │       │               └─ Não: NÓ 2314 | lei = 'N8T7HW55LK5D2ORCKP39'? | bootstrap: 235 recusados, 721 aprovados
│           │       │       │                   ├─ Sim: FOLHA 2328 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 3 aprovados
│           │       │       │                   └─ Não: NÓ 2315 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 229 recusados, 718 aprovados
│           │       │       │                       ├─ Sim: FOLHA 2327 | previsão: APROVADO (1) | bootstrap: 63 recusados, 77 aprovados
│           │       │       │                       └─ Não: NÓ 2316 | lei = 'FU7RSW4CQQY98A2O7J66'? | bootstrap: 166 recusados, 641 aprovados
│           │       │       │                           ├─ Sim: FOLHA 2326 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
│           │       │       │                           └─ Não: NÓ 2317 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 108.5? | bootstrap: 166 recusados, 633 aprovados
│           │       │       │                               ├─ Sim: NÓ 2318 | lei = 'X05BVSK68TQ7YTOSNR22'? | bootstrap: 138 recusados, 430 aprovados
│           │       │       │                               │   ├─ Sim: FOLHA 2324 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
│           │       │       │                               │   └─ Não: NÓ 2319 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 136 recusados, 423 aprovados
│           │       │       │                               │       ├─ Sim: FOLHA 2323 | previsão: APROVADO (1) | bootstrap: 13 recusados, 71 aprovados
│           │       │       │                               │       └─ Não: NÓ 2320 | debt_to_income_ratio = '45'? | bootstrap: 123 recusados, 352 aprovados
│           │       │       │                               │           ├─ Sim: FOLHA 2322 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
│           │       │       │                               │           └─ Não: FOLHA 2321 | previsão: APROVADO (1) | bootstrap: 123 recusados, 344 aprovados
│           │       │       │                               └─ Não: FOLHA 2325 | previsão: APROVADO (1) | bootstrap: 28 recusados, 203 aprovados
│           │       │       └─ Não: NÓ 2332 | lei = '549300RUKY2I2HQ2AI44'? | bootstrap: 423 recusados, 1456 aprovados
│           │       │           ├─ Sim: FOLHA 2356 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
│           │       │           └─ Não: NÓ 2333 | debt_to_income_ratio = '41'? | bootstrap: 420 recusados, 1450 aprovados
│           │       │               ├─ Sim: FOLHA 2355 | previsão: APROVADO (1) | bootstrap: 5 recusados, 38 aprovados
│           │       │               └─ Não: NÓ 2334 | lei = 'EQTWLK1G7ODGC2MGLV11'? | bootstrap: 415 recusados, 1412 aprovados
│           │       │                   ├─ Sim: FOLHA 2354 | previsão: APROVADO (1) | bootstrap: 5 recusados, 8 aprovados
│           │       │                   └─ Não: NÓ 2335 | total_units = '1'? | bootstrap: 410 recusados, 1404 aprovados
│           │       │                       ├─ Sim: NÓ 2337 | income (renda anual declarada em milhares de USD) <= 48.5? | bootstrap: 394 recusados, 1395 aprovados
│           │       │                       │   ├─ Sim: FOLHA 2338 | previsão: RECUSADO (0) | bootstrap: 33 recusados, 14 aprovados
│           │       │                       │   └─ Não: NÓ 2339 | lei = '549300KER3YVZ3U5RK57'? | bootstrap: 361 recusados, 1381 aprovados
│           │       │                       │       ├─ Sim: FOLHA 2353 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
│           │       │                       │       └─ Não: NÓ 2340 | lei = '549300TN9GZ5XRCOXX47'? | bootstrap: 361 recusados, 1370 aprovados
│           │       │                       │           ├─ Sim: FOLHA 2352 | previsão: RECUSADO (0) | bootstrap: 24 recusados, 21 aprovados
│           │       │                       │           └─ Não: NÓ 2341 | construction_method = '2'? | bootstrap: 337 recusados, 1349 aprovados
│           │       │                       │               ├─ Sim: FOLHA 2351 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
│           │       │                       │               └─ Não: NÓ 2342 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 334 recusados, 1341 aprovados
│           │       │                       │                   ├─ Sim: FOLHA 2350 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 6 aprovados
│           │       │                       │                   └─ Não: NÓ 2343 | lei = '549300H3IZO24NSOO931'? | bootstrap: 315 recusados, 1335 aprovados
│           │       │                       │                       ├─ Sim: FOLHA 2349 | previsão: APROVADO (1) | bootstrap: 7 recusados, 12 aprovados
│           │       │                       │                       └─ Não: NÓ 2344 | lei = 'T68X8LLAQYRNDV034K14'? | bootstrap: 308 recusados, 1323 aprovados
│           │       │                       │                           ├─ Sim: FOLHA 2348 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
│           │       │                       │                           └─ Não: NÓ 2345 | lei = 'P7GC0W65KB5CEIHLDX81'? | bootstrap: 308 recusados, 1315 aprovados
│           │       │                       │                               ├─ Sim: FOLHA 2347 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
│           │       │                       │                               └─ Não: FOLHA 2346 | previsão: APROVADO (1) | bootstrap: 306 recusados, 1306 aprovados
│           │       │                       └─ Não: FOLHA 2336 | previsão: RECUSADO (0) | bootstrap: 16 recusados, 9 aprovados
│           │       └─ Não: NÓ 2174 | lei = '549300VCMRO4ST680C11'? | bootstrap: 4182 recusados, 11786 aprovados
│           │           ├─ Sim: FOLHA 2284 | previsão: RECUSADO (0) | bootstrap: 13 recusados, 5 aprovados
│           │           └─ Não: NÓ 2175 | lei = '549300BRR67NLTQSBT29'? | bootstrap: 4169 recusados, 11781 aprovados
│           │               ├─ Sim: FOLHA 2283 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
│           │               └─ Não: NÓ 2176 | lei = '7H6GLXDRUGQFU57RNE97'? | bootstrap: 4169 recusados, 11766 aprovados
│           │                   ├─ Sim: NÓ 2272 | derived_race = 'White'? | bootstrap: 23 recusados, 350 aprovados
│           │                   │   ├─ Sim: NÓ 2274 | debt_to_income_ratio = '50%-60%'? | bootstrap: 22 recusados, 222 aprovados
│           │                   │   │   ├─ Sim: FOLHA 2282 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 0 aprovados
│           │                   │   │   └─ Não: NÓ 2275 | debt_to_income_ratio = '>60%'? | bootstrap: 15 recusados, 222 aprovados
│           │                   │   │       ├─ Sim: FOLHA 2281 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 0 aprovados
│           │                   │   │       └─ Não: NÓ 2276 | debt_to_income_ratio = '43'? | bootstrap: 8 recusados, 222 aprovados
│           │                   │   │           ├─ Sim: FOLHA 2280 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
│           │                   │   │           └─ Não: NÓ 2277 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 8 recusados, 210 aprovados
│           │                   │   │               ├─ Sim: FOLHA 2279 | previsão: APROVADO (1) | bootstrap: 4 recusados, 26 aprovados
│           │                   │   │               └─ Não: FOLHA 2278 | previsão: APROVADO (1) | bootstrap: 4 recusados, 184 aprovados
│           │                   │   └─ Não: FOLHA 2273 | previsão: APROVADO (1) | bootstrap: 1 recusados, 128 aprovados
│           │                   └─ Não: NÓ 2177 | lei = '549300FGXN1K3HLB1R50'? | bootstrap: 4146 recusados, 11416 aprovados
│           │                       ├─ Sim: NÓ 2233 | debt_to_income_ratio = '<20%'? | bootstrap: 109 recusados, 519 aprovados
│           │                       │   ├─ Sim: NÓ 2269 | loan_type = '1'? | bootstrap: 10 recusados, 61 aprovados
│           │                       │   │   ├─ Sim: FOLHA 2271 | previsão: APROVADO (1) | bootstrap: 7 recusados, 54 aprovados
│           │                       │   │   └─ Não: FOLHA 2270 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
│           │                       │   └─ Não: NÓ 2234 | loan_type = '2'? | bootstrap: 99 recusados, 458 aprovados
│           │                       │       ├─ Sim: NÓ 2258 | debt_to_income_ratio = '48'? | bootstrap: 53 recusados, 99 aprovados
│           │                       │       │   ├─ Sim: FOLHA 2268 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
│           │                       │       │   └─ Não: NÓ 2259 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 52 recusados, 94 aprovados
│           │                       │       │       ├─ Sim: FOLHA 2267 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
│           │                       │       │       └─ Não: NÓ 2260 | derived_race = 'Asian'? | bootstrap: 51 recusados, 86 aprovados
│           │                       │       │           ├─ Sim: FOLHA 2266 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
│           │                       │       │           └─ Não: NÓ 2261 | loan_amount (valor solicitado em USD) <= 280000.0? | bootstrap: 49 recusados, 82 aprovados
│           │                       │       │               ├─ Sim: NÓ 2262 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 48 recusados, 69 aprovados
│           │                       │       │               │   ├─ Sim: FOLHA 2264 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 1 aprovados
│           │                       │       │               │   └─ Não: FOLHA 2263 | previsão: APROVADO (1) | bootstrap: 40 recusados, 68 aprovados
│           │                       │       │               └─ Não: FOLHA 2265 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
│           │                       │       └─ Não: NÓ 2235 | loan_amount (valor solicitado em USD) <= 150000.0? | bootstrap: 46 recusados, 359 aprovados
│           │                       │           ├─ Sim: FOLHA 2236 | previsão: APROVADO (1) | bootstrap: 13 recusados, 38 aprovados
│           │                       │           └─ Não: NÓ 2237 | derived_ethnicity = 'Joint'? | bootstrap: 33 recusados, 321 aprovados
│           │                       │               ├─ Sim: FOLHA 2257 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
│           │                       │               └─ Não: NÓ 2238 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 33 recusados, 309 aprovados
│           │                       │                   ├─ Sim: NÓ 2254 | loan_term (prazo em meses) <= 210.0? | bootstrap: 2 recusados, 67 aprovados
│           │                       │                   │   ├─ Sim: FOLHA 2255 | previsão: APROVADO (1) | bootstrap: 0 recusados, 45 aprovados
│           │                       │                   │   └─ Não: FOLHA 2256 | previsão: APROVADO (1) | bootstrap: 2 recusados, 22 aprovados
│           │                       │                   └─ Não: NÓ 2239 | debt_to_income_ratio = '39'? | bootstrap: 31 recusados, 242 aprovados
│           │                       │                       ├─ Sim: NÓ 2251 | income (renda anual declarada em milhares de USD) <= 133.0? | bootstrap: 5 recusados, 22 aprovados
│           │                       │                       │   ├─ Sim: FOLHA 2252 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
│           │                       │                       │   └─ Não: FOLHA 2253 | previsão: APROVADO (1) | bootstrap: 2 recusados, 16 aprovados
│           │                       │                       └─ Não: NÓ 2240 | derived_race = 'White'? | bootstrap: 26 recusados, 220 aprovados
│           │                       │                           ├─ Sim: NÓ 2244 | loan_type = '3'? | bootstrap: 20 recusados, 135 aprovados
│           │                       │                           │   ├─ Sim: NÓ 2246 | loan_term (prazo em meses) <= 314.5? | bootstrap: 3 recusados, 26 aprovados
│           │                       │                           │   │   ├─ Sim: FOLHA 2247 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
│           │                       │                           │   │   └─ Não: NÓ 2248 | loan_amount (valor solicitado em USD) <= 440000.0? | bootstrap: 1 recusados, 20 aprovados
│           │                       │                           │   │       ├─ Sim: FOLHA 2249 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
│           │                       │                           │   │       └─ Não: FOLHA 2250 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
│           │                       │                           │   └─ Não: FOLHA 2245 | previsão: APROVADO (1) | bootstrap: 17 recusados, 109 aprovados
│           │                       │                           └─ Não: NÓ 2241 | property_value (valor do imóvel em USD) <= 1220000.0? | bootstrap: 6 recusados, 85 aprovados
│           │                       │                               ├─ Sim: FOLHA 2242 | previsão: APROVADO (1) | bootstrap: 5 recusados, 80 aprovados
│           │                       │                               └─ Não: FOLHA 2243 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
│           │                       └─ Não: NÓ 2178 | lei = '549300RPOGWJRH63HS39'? | bootstrap: 4037 recusados, 10897 aprovados
│           │                           ├─ Sim: FOLHA 2232 | previsão: APROVADO (1) | bootstrap: 0 recusados, 45 aprovados
│           │                           └─ Não: NÓ 2179 | lei = '549300U3721PJGQZYY68'? | bootstrap: 4037 recusados, 10852 aprovados
│           │                               ├─ Sim: NÓ 2229 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 7 recusados, 87 aprovados
│           │                               │   ├─ Sim: FOLHA 2231 | previsão: APROVADO (1) | bootstrap: 1 recusados, 19 aprovados
│           │                               │   └─ Não: FOLHA 2230 | previsão: APROVADO (1) | bootstrap: 6 recusados, 68 aprovados
│           │                               └─ Não: NÓ 2180 | construction_method = '1'? | bootstrap: 4030 recusados, 10765 aprovados
│           │                                   ├─ Sim: NÓ 2202 | lei = '549300ESRQ6OLBB56N04'? | bootstrap: 900 recusados, 8281 aprovados
│           │                                   │   ├─ Sim: FOLHA 2228 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
│           │                                   │   └─ Não: NÓ 2203 | lei = '549300KM40FP4MSQU941'? | bootstrap: 898 recusados, 8275 aprovados
│           │                                   │       ├─ Sim: FOLHA 2227 | previsão: RECUSADO (0) | bootstrap: 13 recusados, 11 aprovados
│           │                                   │       └─ Não: NÓ 2204 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 885 recusados, 8264 aprovados
│           │                                   │           ├─ Sim: FOLHA 2226 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 11 aprovados
│           │                                   │           └─ Não: NÓ 2205 | loan_type = '2'? | bootstrap: 873 recusados, 8253 aprovados
│           │                                   │               ├─ Sim: NÓ 2219 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 103 recusados, 483 aprovados
│           │                                   │               │   ├─ Sim: FOLHA 2225 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
│           │                                   │               │   └─ Não: NÓ 2220 | lei = '254900ZFWS2106HWPH46'? | bootstrap: 103 recusados, 455 aprovados
│           │                                   │               │       ├─ Sim: FOLHA 2224 | previsão: APROVADO (1) | bootstrap: 0 recusados, 21 aprovados
│           │                                   │               │       └─ Não: NÓ 2221 | debt_to_income_ratio = '>60%'? | bootstrap: 103 recusados, 434 aprovados
│           │                                   │               │           ├─ Sim: FOLHA 2223 | previsão: RECUSADO (0) | bootstrap: 32 recusados, 0 aprovados
│           │                                   │               │           └─ Não: FOLHA 2222 | previsão: APROVADO (1) | bootstrap: 71 recusados, 434 aprovados
│           │                                   │               └─ Não: NÓ 2206 | lei = '549300FX7K8PTEQUU487'? | bootstrap: 770 recusados, 7770 aprovados
│           │                                   │                   ├─ Sim: NÓ 2214 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 138.0? | bootstrap: 10 recusados, 13 aprovados
│           │                                   │                   │   ├─ Sim: NÓ 2215 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 110.5? | bootstrap: 8 recusados, 7 aprovados
│           │                                   │                   │   │   ├─ Sim: FOLHA 2216 | previsão: APROVADO (1) | bootstrap: 4 recusados, 6 aprovados
│           │                                   │                   │   │   └─ Não: FOLHA 2217 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 1 aprovados
│           │                                   │                   │   └─ Não: FOLHA 2218 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
│           │                                   │                   └─ Não: NÓ 2207 | debt_to_income_ratio = '37'? | bootstrap: 760 recusados, 7757 aprovados
│           │                                   │                       ├─ Sim: NÓ 2211 | property_value (valor do imóvel em USD) <= 230000.0? | bootstrap: 9 recusados, 209 aprovados
│           │                                   │                       │   ├─ Sim: FOLHA 2212 | previsão: APROVADO (1) | bootstrap: 4 recusados, 24 aprovados
│           │                                   │                       │   └─ Não: FOLHA 2213 | previsão: APROVADO (1) | bootstrap: 5 recusados, 185 aprovados
│           │                                   │                       └─ Não: NÓ 2208 | lei = '54930001NSTOD85LT125'? | bootstrap: 751 recusados, 7548 aprovados
│           │                                   │                           ├─ Sim: FOLHA 2210 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
│           │                                   │                           └─ Não: FOLHA 2209 | previsão: APROVADO (1) | bootstrap: 751 recusados, 7523 aprovados
│           │                                   └─ Não: NÓ 2181 | lei = '549300Q3ROEQU3D1IW91'? | bootstrap: 3130 recusados, 2484 aprovados
│           │                                       ├─ Sim: NÓ 2183 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 396 recusados, 193 aprovados
│           │                                       │   ├─ Sim: NÓ 2191 | debt_to_income_ratio = '40'? | bootstrap: 302 recusados, 158 aprovados
│           │                                       │   │   ├─ Sim: FOLHA 2201 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
│           │                                       │   │   └─ Não: NÓ 2192 | debt_to_income_ratio = '42'? | bootstrap: 299 recusados, 152 aprovados
│           │                                       │   │       ├─ Sim: FOLHA 2200 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 5 aprovados
│           │                                       │   │       └─ Não: NÓ 2193 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 291 recusados, 147 aprovados
│           │                                       │   │           ├─ Sim: FOLHA 2199 | previsão: APROVADO (1) | bootstrap: 40 recusados, 45 aprovados
│           │                                       │   │           └─ Não: NÓ 2194 | debt_to_income_ratio = '43'? | bootstrap: 251 recusados, 102 aprovados
│           │                                       │   │               ├─ Sim: FOLHA 2198 | previsão: APROVADO (1) | bootstrap: 6 recusados, 12 aprovados
│           │                                       │   │               └─ Não: NÓ 2195 | debt_to_income_ratio = '<20%'? | bootstrap: 245 recusados, 90 aprovados
│           │                                       │   │                   ├─ Sim: FOLHA 2197 | previsão: RECUSADO (0) | bootstrap: 32 recusados, 7 aprovados
│           │                                       │   │                   └─ Não: FOLHA 2196 | previsão: RECUSADO (0) | bootstrap: 213 recusados, 83 aprovados
│           │                                       │   └─ Não: NÓ 2184 | loan_amount (valor solicitado em USD) <= 120000.0? | bootstrap: 94 recusados, 35 aprovados
│           │                                       │       ├─ Sim: FOLHA 2185 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 6 aprovados
│           │                                       │       └─ Não: NÓ 2186 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 88 recusados, 29 aprovados
│           │                                       │           ├─ Sim: NÓ 2188 | loan_amount (valor solicitado em USD) <= 150000.0? | bootstrap: 12 recusados, 6 aprovados
│           │                                       │           │   ├─ Sim: FOLHA 2189 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
│           │                                       │           │   └─ Não: FOLHA 2190 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 1 aprovados
│           │                                       │           └─ Não: FOLHA 2187 | previsão: RECUSADO (0) | bootstrap: 76 recusados, 23 aprovados
│           │                                       └─ Não: FOLHA 2182 | previsão: RECUSADO (0) | bootstrap: 2734 recusados, 2291 aprovados
│           └─ Não: NÓ 2359 | lei = 'B4TYDEB6GKMZO031MB27'? | bootstrap: 23441 recusados, 211759 aprovados
│               ├─ Sim: NÓ 2625 | loan_amount (valor solicitado em USD) <= 130000.0? | bootstrap: 750 recusados, 2149 aprovados
│               │   ├─ Sim: NÓ 2626 | derived_ethnicity = 'Joint'? | bootstrap: 179 recusados, 116 aprovados
│               │   │   ├─ Sim: FOLHA 2644 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 2 aprovados
│               │   │   └─ Não: NÓ 2627 | property_value (valor do imóvel em USD) <= 150000.0? | bootstrap: 171 recusados, 114 aprovados
│               │   │       ├─ Sim: NÓ 2628 | derived_race = 'White'? | bootstrap: 4 recusados, 15 aprovados
│               │   │       │   ├─ Sim: FOLHA 2630 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
│               │   │       │   └─ Não: FOLHA 2629 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
│               │   │       └─ Não: NÓ 2631 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 167 recusados, 99 aprovados
│               │   │           ├─ Sim: FOLHA 2643 | previsão: RECUSADO (0) | bootstrap: 56 recusados, 24 aprovados
│               │   │           └─ Não: NÓ 2632 | income (renda anual declarada em milhares de USD) <= 91.5? | bootstrap: 111 recusados, 75 aprovados
│               │   │               ├─ Sim: FOLHA 2633 | previsão: RECUSADO (0) | bootstrap: 57 recusados, 15 aprovados
│               │   │               └─ Não: NÓ 2634 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 54 recusados, 60 aprovados
│               │   │                   ├─ Sim: NÓ 2640 | loan_amount (valor solicitado em USD) <= 80000.0? | bootstrap: 6 recusados, 19 aprovados
│               │   │                   │   ├─ Sim: FOLHA 2641 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
│               │   │                   │   └─ Não: FOLHA 2642 | previsão: APROVADO (1) | bootstrap: 1 recusados, 15 aprovados
│               │   │                   └─ Não: NÓ 2635 | loan_amount (valor solicitado em USD) <= 110000.0? | bootstrap: 48 recusados, 41 aprovados
│               │   │                       ├─ Sim: NÓ 2636 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 42 recusados, 33 aprovados
│               │   │                       │   ├─ Sim: FOLHA 2638 | previsão: APROVADO (1) | bootstrap: 11 recusados, 13 aprovados
│               │   │                       │   └─ Não: FOLHA 2637 | previsão: RECUSADO (0) | bootstrap: 31 recusados, 20 aprovados
│               │   │                       └─ Não: FOLHA 2639 | previsão: APROVADO (1) | bootstrap: 6 recusados, 8 aprovados
│               │   └─ Não: NÓ 2645 | debt_to_income_ratio = '>60%'? | bootstrap: 571 recusados, 2033 aprovados
│               │       ├─ Sim: FOLHA 2669 | previsão: RECUSADO (0) | bootstrap: 211 recusados, 3 aprovados
│               │       └─ Não: NÓ 2646 | debt_to_income_ratio = '44'? | bootstrap: 360 recusados, 2030 aprovados
│               │           ├─ Sim: NÓ 2662 | derived_race = 'Asian'? | bootstrap: 13 recusados, 130 aprovados
│               │           │   ├─ Sim: NÓ 2666 | income (renda anual declarada em milhares de USD) <= 94.5? | bootstrap: 1 recusados, 49 aprovados
│               │           │   │   ├─ Sim: FOLHA 2667 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
│               │           │   │   └─ Não: FOLHA 2668 | previsão: APROVADO (1) | bootstrap: 0 recusados, 43 aprovados
│               │           │   └─ Não: NÓ 2663 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 65.0? | bootstrap: 12 recusados, 81 aprovados
│               │           │       ├─ Sim: FOLHA 2664 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
│               │           │       └─ Não: FOLHA 2665 | previsão: APROVADO (1) | bootstrap: 9 recusados, 76 aprovados
│               │           └─ Não: NÓ 2647 | derived_race = 'Black or African American'? | bootstrap: 347 recusados, 1900 aprovados
│               │               ├─ Sim: FOLHA 2661 | previsão: APROVADO (1) | bootstrap: 77 recusados, 314 aprovados
│               │               └─ Não: NÓ 2648 | debt_to_income_ratio = '41'? | bootstrap: 270 recusados, 1586 aprovados
│               │                   ├─ Sim: NÓ 2652 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 85.5? | bootstrap: 8 recusados, 100 aprovados
│               │                   │   ├─ Sim: FOLHA 2653 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
│               │                   │   └─ Não: NÓ 2654 | derived_race = 'Asian'? | bootstrap: 5 recusados, 91 aprovados
│               │                   │       ├─ Sim: NÓ 2658 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 146.5? | bootstrap: 3 recusados, 33 aprovados
│               │                   │       │   ├─ Sim: FOLHA 2659 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
│               │                   │       │   └─ Não: FOLHA 2660 | previsão: APROVADO (1) | bootstrap: 3 recusados, 14 aprovados
│               │                   │       └─ Não: NÓ 2655 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 109.0? | bootstrap: 2 recusados, 58 aprovados
│               │                   │           ├─ Sim: FOLHA 2656 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
│               │                   │           └─ Não: FOLHA 2657 | previsão: APROVADO (1) | bootstrap: 1 recusados, 44 aprovados
│               │                   └─ Não: NÓ 2649 | debt_to_income_ratio = '42'? | bootstrap: 262 recusados, 1486 aprovados
│               │                       ├─ Sim: FOLHA 2651 | previsão: APROVADO (1) | bootstrap: 6 recusados, 108 aprovados
│               │                       └─ Não: FOLHA 2650 | previsão: APROVADO (1) | bootstrap: 256 recusados, 1378 aprovados
│               └─ Não: NÓ 2360 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 22691 recusados, 209610 aprovados
│                   ├─ Sim: FOLHA 2624 | previsão: APROVADO (1) | bootstrap: 89 recusados, 517 aprovados
│                   └─ Não: NÓ 2361 | loan_term (prazo em meses) <= 365.5? | bootstrap: 22602 recusados, 209093 aprovados
│                       ├─ Sim: NÓ 2362 | lei = '549300VZVN841I2ILS84'? | bootstrap: 22031 recusados, 206546 aprovados
│                       │   ├─ Sim: NÓ 2580 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 203 recusados, 5546 aprovados
│                       │   │   ├─ Sim: NÓ 2582 | debt_to_income_ratio = '<20%'? | bootstrap: 37 recusados, 840 aprovados
│                       │   │   │   ├─ Sim: FOLHA 2606 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
│                       │   │   │   └─ Não: NÓ 2583 | derived_race = 'Joint'? | bootstrap: 37 recusados, 818 aprovados
│                       │   │   │       ├─ Sim: FOLHA 2605 | previsão: APROVADO (1) | bootstrap: 2 recusados, 11 aprovados
│                       │   │   │       └─ Não: NÓ 2584 | debt_to_income_ratio = '44'? | bootstrap: 35 recusados, 807 aprovados
│                       │   │   │           ├─ Sim: NÓ 2602 | property_value (valor do imóvel em USD) <= 305000.0? | bootstrap: 2 recusados, 46 aprovados
│                       │   │   │           │   ├─ Sim: FOLHA 2603 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
│                       │   │   │           │   └─ Não: FOLHA 2604 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
│                       │   │   │           └─ Não: NÓ 2585 | construction_method = '1'? | bootstrap: 33 recusados, 761 aprovados
│                       │   │   │               ├─ Sim: NÓ 2587 | property_value (valor do imóvel em USD) <= 270000.0? | bootstrap: 30 recusados, 751 aprovados
│                       │   │   │               │   ├─ Sim: NÓ 2588 | loan_type = '3'? | bootstrap: 2 recusados, 191 aprovados
│                       │   │   │               │   │   ├─ Sim: FOLHA 2592 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
│                       │   │   │               │   │   └─ Não: NÓ 2589 | property_value (valor do imóvel em USD) <= 220000.0? | bootstrap: 2 recusados, 183 aprovados
│                       │   │   │               │   │       ├─ Sim: FOLHA 2590 | previsão: APROVADO (1) | bootstrap: 2 recusados, 93 aprovados
│                       │   │   │               │   │       └─ Não: FOLHA 2591 | previsão: APROVADO (1) | bootstrap: 0 recusados, 90 aprovados
│                       │   │   │               │   └─ Não: NÓ 2593 | debt_to_income_ratio = '40'? | bootstrap: 28 recusados, 560 aprovados
│                       │   │   │               │       ├─ Sim: FOLHA 2601 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
│                       │   │   │               │       └─ Não: NÓ 2594 | loan_amount (valor solicitado em USD) <= 360000.0? | bootstrap: 28 recusados, 538 aprovados
│                       │   │   │               │           ├─ Sim: FOLHA 2595 | previsão: APROVADO (1) | bootstrap: 19 recusados, 226 aprovados
│                       │   │   │               │           └─ Não: NÓ 2596 | debt_to_income_ratio = '41'? | bootstrap: 9 recusados, 312 aprovados
│                       │   │   │               │               ├─ Sim: FOLHA 2600 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
│                       │   │   │               │               └─ Não: NÓ 2597 | debt_to_income_ratio = '47'? | bootstrap: 9 recusados, 298 aprovados
│                       │   │   │               │                   ├─ Sim: FOLHA 2599 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
│                       │   │   │               │                   └─ Não: FOLHA 2598 | previsão: APROVADO (1) | bootstrap: 9 recusados, 288 aprovados
│                       │   │   │               └─ Não: FOLHA 2586 | previsão: APROVADO (1) | bootstrap: 3 recusados, 10 aprovados
│                       │   │   └─ Não: FOLHA 2581 | previsão: APROVADO (1) | bootstrap: 166 recusados, 4706 aprovados
│                       │   └─ Não: NÓ 2363 | debt_to_income_ratio = '39'? | bootstrap: 21828 recusados, 201000 aprovados
│                       │       ├─ Sim: NÓ 2545 | lei = '549300121SF0K2LN2804'? | bootstrap: 383 recusados, 7063 aprovados
│                       │       │   ├─ Sim: FOLHA 2579 | previsão: APROVADO (1) | bootstrap: 0 recusados, 66 aprovados
│                       │       │   └─ Não: NÓ 2546 | lei = '549300OPCWU6E72WUT29'? | bootstrap: 383 recusados, 6997 aprovados
│                       │       │       ├─ Sim: FOLHA 2578 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
│                       │       │       └─ Não: NÓ 2547 | lei = 'KB1H1DSPRFMYMCUFXT09'? | bootstrap: 380 recusados, 6985 aprovados
│                       │       │           ├─ Sim: FOLHA 2577 | previsão: APROVADO (1) | bootstrap: 7 recusados, 71 aprovados
│                       │       │           └─ Não: NÓ 2548 | lei = '549300157R33LDICP344'? | bootstrap: 373 recusados, 6914 aprovados
│                       │       │               ├─ Sim: FOLHA 2576 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
│                       │       │               └─ Não: NÓ 2549 | lei = '5493000NYUJT9UC6G261'? | bootstrap: 368 recusados, 6909 aprovados
│                       │       │                   ├─ Sim: FOLHA 2575 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
│                       │       │                   └─ Não: NÓ 2550 | lei = '549300PL8ER6H23P0Z91'? | bootstrap: 365 recusados, 6902 aprovados
│                       │       │                       ├─ Sim: FOLHA 2574 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
│                       │       │                       └─ Não: NÓ 2551 | lei = '5493004N9PMBSLEZOF16'? | bootstrap: 365 recusados, 6877 aprovados
│                       │       │                           ├─ Sim: FOLHA 2573 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
│                       │       │                           └─ Não: NÓ 2552 | lei = '549300AQ3T62GXDU7D76'? | bootstrap: 363 recusados, 6868 aprovados
│                       │       │                               ├─ Sim: NÓ 2562 | loan_type = '1'? | bootstrap: 2 recusados, 135 aprovados
│                       │       │                               │   ├─ Sim: NÓ 2568 | property_value (valor do imóvel em USD) <= 740000.0? | bootstrap: 1 recusados, 78 aprovados
│                       │       │                               │   │   ├─ Sim: FOLHA 2569 | previsão: APROVADO (1) | bootstrap: 0 recusados, 62 aprovados
│                       │       │                               │   │   └─ Não: NÓ 2570 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 130.0? | bootstrap: 1 recusados, 16 aprovados
│                       │       │                               │   │       ├─ Sim: FOLHA 2571 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
│                       │       │                               │   │       └─ Não: FOLHA 2572 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
│                       │       │                               │   └─ Não: NÓ 2563 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 1 recusados, 57 aprovados
│                       │       │                               │       ├─ Sim: NÓ 2565 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 67.0? | bootstrap: 1 recusados, 46 aprovados
│                       │       │                               │       │   ├─ Sim: FOLHA 2566 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
│                       │       │                               │       │   └─ Não: FOLHA 2567 | previsão: APROVADO (1) | bootstrap: 0 recusados, 37 aprovados
│                       │       │                               │       └─ Não: FOLHA 2564 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
│                       │       │                               └─ Não: NÓ 2553 | lei = '254900DTLHVWQ7NP7R34'? | bootstrap: 361 recusados, 6733 aprovados
│                       │       │                                   ├─ Sim: NÓ 2559 | income (renda anual declarada em milhares de USD) <= 113.0? | bootstrap: 3 recusados, 29 aprovados
│                       │       │                                   │   ├─ Sim: FOLHA 2560 | previsão: APROVADO (1) | bootstrap: 3 recusados, 10 aprovados
│                       │       │                                   │   └─ Não: FOLHA 2561 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
│                       │       │                                   └─ Não: NÓ 2554 | construction_method = '2'? | bootstrap: 358 recusados, 6704 aprovados
│                       │       │                                       ├─ Sim: FOLHA 2558 | previsão: APROVADO (1) | bootstrap: 16 recusados, 96 aprovados
│                       │       │                                       └─ Não: NÓ 2555 | lei = '549300N3JBQ15RQ7AV97'? | bootstrap: 342 recusados, 6608 aprovados
│                       │       │                                           ├─ Sim: FOLHA 2557 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
│                       │       │                                           └─ Não: FOLHA 2556 | previsão: APROVADO (1) | bootstrap: 340 recusados, 6603 aprovados
│                       │       └─ Não: NÓ 2364 | total_units = '3'? | bootstrap: 21445 recusados, 193937 aprovados
│                       │           ├─ Sim: NÓ 2530 | derived_race = 'Black or African American'? | bootstrap: 121 recusados, 307 aprovados
│                       │           │   ├─ Sim: NÓ 2536 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 40 recusados, 80 aprovados
│                       │           │   │   ├─ Sim: FOLHA 2544 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
│                       │           │   │   └─ Não: NÓ 2537 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 64.5? | bootstrap: 39 recusados, 74 aprovados
│                       │           │   │       ├─ Sim: FOLHA 2538 | previsão: APROVADO (1) | bootstrap: 17 recusados, 52 aprovados
│                       │           │   │       └─ Não: NÓ 2539 | loan_type = '1'? | bootstrap: 22 recusados, 22 aprovados
│                       │           │   │           ├─ Sim: FOLHA 2543 | previsão: APROVADO (1) | bootstrap: 5 recusados, 10 aprovados
│                       │           │   │           └─ Não: NÓ 2540 | property_value (valor do imóvel em USD) <= 545000.0? | bootstrap: 17 recusados, 12 aprovados
│                       │           │   │               ├─ Sim: FOLHA 2541 | previsão: APROVADO (1) | bootstrap: 6 recusados, 9 aprovados
│                       │           │   │               └─ Não: FOLHA 2542 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 3 aprovados
│                       │           │   └─ Não: NÓ 2531 | debt_to_income_ratio = '48'? | bootstrap: 81 recusados, 227 aprovados
│                       │           │       ├─ Sim: FOLHA 2535 | previsão: APROVADO (1) | bootstrap: 2 recusados, 17 aprovados
│                       │           │       └─ Não: NÓ 2532 | debt_to_income_ratio = '44'? | bootstrap: 79 recusados, 210 aprovados
│                       │           │           ├─ Sim: FOLHA 2534 | previsão: APROVADO (1) | bootstrap: 4 recusados, 11 aprovados
│                       │           │           └─ Não: FOLHA 2533 | previsão: APROVADO (1) | bootstrap: 75 recusados, 199 aprovados
│                       │           └─ Não: NÓ 2365 | derived_race = 'White'? | bootstrap: 21324 recusados, 193630 aprovados
│                       │               ├─ Sim: NÓ 2427 | construction_method = '2'? | bootstrap: 13427 recusados, 142887 aprovados
│                       │               │   ├─ Sim: NÓ 2509 | lei = '549300FGXN1K3HLB1R50'? | bootstrap: 730 recusados, 3295 aprovados
│                       │               │   │   ├─ Sim: NÓ 2523 | property_value (valor do imóvel em USD) <= 390000.0? | bootstrap: 66 recusados, 159 aprovados
│                       │               │   │   │   ├─ Sim: NÓ 2524 | loan_type = '3'? | bootstrap: 54 recusados, 146 aprovados
│                       │               │   │   │   │   ├─ Sim: FOLHA 2526 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
│                       │               │   │   │   │   └─ Não: FOLHA 2525 | previsão: APROVADO (1) | bootstrap: 53 recusados, 140 aprovados
│                       │               │   │   │   └─ Não: NÓ 2527 | loan_amount (valor solicitado em USD) <= 360000.0? | bootstrap: 12 recusados, 13 aprovados
│                       │               │   │   │       ├─ Sim: FOLHA 2528 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 3 aprovados
│                       │               │   │   │       └─ Não: FOLHA 2529 | previsão: APROVADO (1) | bootstrap: 4 recusados, 10 aprovados
│                       │               │   │   └─ Não: NÓ 2510 | lei = '54930034MNPILHP25H80'? | bootstrap: 664 recusados, 3136 aprovados
│                       │               │   │       ├─ Sim: FOLHA 2522 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
│                       │               │   │       └─ Não: NÓ 2511 | debt_to_income_ratio = '46'? | bootstrap: 664 recusados, 3119 aprovados
│                       │               │   │           ├─ Sim: FOLHA 2521 | previsão: APROVADO (1) | bootstrap: 8 recusados, 162 aprovados
│                       │               │   │           └─ Não: NÓ 2512 | debt_to_income_ratio = '>60%'? | bootstrap: 656 recusados, 2957 aprovados
│                       │               │   │               ├─ Sim: NÓ 2518 | property_value (valor do imóvel em USD) <= 410000.0? | bootstrap: 133 recusados, 21 aprovados
│                       │               │   │               │   ├─ Sim: FOLHA 2519 | previsão: RECUSADO (0) | bootstrap: 119 recusados, 13 aprovados
│                       │               │   │               │   └─ Não: FOLHA 2520 | previsão: RECUSADO (0) | bootstrap: 14 recusados, 8 aprovados
│                       │               │   │               └─ Não: NÓ 2513 | debt_to_income_ratio = '<20%'? | bootstrap: 523 recusados, 2936 aprovados
│                       │               │   │                   ├─ Sim: FOLHA 2517 | previsão: APROVADO (1) | bootstrap: 27 recusados, 66 aprovados
│                       │               │   │                   └─ Não: NÓ 2514 | lei = '549300ZBXINP1Y94CN02'? | bootstrap: 496 recusados, 2870 aprovados
│                       │               │   │                       ├─ Sim: FOLHA 2516 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
│                       │               │   │                       └─ Não: FOLHA 2515 | previsão: APROVADO (1) | bootstrap: 496 recusados, 2861 aprovados
│                       │               │   └─ Não: NÓ 2428 | lei = '549300JYXTZDSPJEPI44'? | bootstrap: 12697 recusados, 139592 aprovados
│                       │               │       ├─ Sim: NÓ 2494 | debt_to_income_ratio = '>60%'? | bootstrap: 142 recusados, 521 aprovados
│                       │               │       │   ├─ Sim: NÓ 2506 | loan_amount (valor solicitado em USD) <= 415000.0? | bootstrap: 12 recusados, 1 aprovados
│                       │               │       │   │   ├─ Sim: FOLHA 2507 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 0 aprovados
│                       │               │       │   │   └─ Não: FOLHA 2508 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 1 aprovados
│                       │               │       │   └─ Não: NÓ 2495 | debt_to_income_ratio = '49'? | bootstrap: 130 recusados, 520 aprovados
│                       │               │       │       ├─ Sim: NÓ 2497 | loan_type = '1'? | bootstrap: 9 recusados, 57 aprovados
│                       │               │       │       │   ├─ Sim: NÓ 2499 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 136.5? | bootstrap: 9 recusados, 31 aprovados
│                       │               │       │       │   │   ├─ Sim: NÓ 2500 | income (renda anual declarada em milhares de USD) <= 68.5? | bootstrap: 3 recusados, 23 aprovados
│                       │               │       │       │   │   │   ├─ Sim: FOLHA 2501 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
│                       │               │       │       │   │   │   └─ Não: NÓ 2502 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 113.0? | bootstrap: 3 recusados, 15 aprovados
│                       │               │       │       │   │   │       ├─ Sim: FOLHA 2503 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
│                       │               │       │       │   │   │       └─ Não: FOLHA 2504 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
│                       │               │       │       │   │   └─ Não: FOLHA 2505 | previsão: APROVADO (1) | bootstrap: 6 recusados, 8 aprovados
│                       │               │       │       │   └─ Não: FOLHA 2498 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
│                       │               │       │       └─ Não: FOLHA 2496 | previsão: APROVADO (1) | bootstrap: 121 recusados, 463 aprovados
│                       │               │       └─ Não: NÓ 2429 | debt_to_income_ratio = '50%-60%'? | bootstrap: 12555 recusados, 139071 aprovados
│                       │               │           ├─ Sim: NÓ 2469 | lei = 'AD6GFRVSDT01YPT1CS68'? | bootstrap: 2631 recusados, 14047 aprovados
│                       │               │           │   ├─ Sim: FOLHA 2493 | previsão: RECUSADO (0) | bootstrap: 16 recusados, 7 aprovados
│                       │               │           │   └─ Não: NÓ 2470 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 2615 recusados, 14040 aprovados
│                       │               │           │       ├─ Sim: NÓ 2482 | lei = 'KB1H1DSPRFMYMCUFXT09'? | bootstrap: 1017 recusados, 4575 aprovados
│                       │               │           │       │   ├─ Sim: FOLHA 2492 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 6 aprovados
│                       │               │           │       │   └─ Não: NÓ 2483 | loan_amount (valor solicitado em USD) <= 940000.0? | bootstrap: 1009 recusados, 4569 aprovados
│                       │               │           │       │       ├─ Sim: NÓ 2484 | lei = '549300FNXYY540N23N64'? | bootstrap: 987 recusados, 4551 aprovados
│                       │               │           │       │       │   ├─ Sim: NÓ 2488 | loan_amount (valor solicitado em USD) <= 520000.0? | bootstrap: 17 recusados, 36 aprovados
│                       │               │           │       │       │   │   ├─ Sim: FOLHA 2489 | previsão: APROVADO (1) | bootstrap: 17 recusados, 27 aprovados
│                       │               │           │       │       │   │   └─ Não: FOLHA 2490 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
│                       │               │           │       │       │   └─ Não: NÓ 2485 | lei = '254900WEP08K7U7S3A80'? | bootstrap: 970 recusados, 4515 aprovados
│                       │               │           │       │       │       ├─ Sim: FOLHA 2487 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
│                       │               │           │       │       │       └─ Não: FOLHA 2486 | previsão: APROVADO (1) | bootstrap: 970 recusados, 4499 aprovados
│                       │               │           │       │       └─ Não: FOLHA 2491 | previsão: RECUSADO (0) | bootstrap: 22 recusados, 18 aprovados
│                       │               │           │       └─ Não: NÓ 2471 | lei = '5493008NWHQT1R22C024'? | bootstrap: 1598 recusados, 9465 aprovados
│                       │               │           │           ├─ Sim: FOLHA 2481 | previsão: APROVADO (1) | bootstrap: 0 recusados, 39 aprovados
│                       │               │           │           └─ Não: NÓ 2472 | lei = '03D0JEWFDFUS0SEEKG89'? | bootstrap: 1598 recusados, 9426 aprovados
│                       │               │           │               ├─ Sim: FOLHA 2480 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 3 aprovados
│                       │               │           │               └─ Não: NÓ 2473 | lei = '549300KBWX4NV5Q1E376'? | bootstrap: 1588 recusados, 9423 aprovados
│                       │               │           │                   ├─ Sim: NÓ 2477 | loan_amount (valor solicitado em USD) <= 265000.0? | bootstrap: 25 recusados, 20 aprovados
│                       │               │           │                   │   ├─ Sim: FOLHA 2478 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 4 aprovados
│                       │               │           │                   │   └─ Não: FOLHA 2479 | previsão: APROVADO (1) | bootstrap: 14 recusados, 16 aprovados
│                       │               │           │                   └─ Não: NÓ 2474 | lei = '549300U3721PJGQZYY68'? | bootstrap: 1563 recusados, 9403 aprovados
│                       │               │           │                       ├─ Sim: FOLHA 2476 | previsão: APROVADO (1) | bootstrap: 17 recusados, 155 aprovados
│                       │               │           │                       └─ Não: FOLHA 2475 | previsão: APROVADO (1) | bootstrap: 1546 recusados, 9248 aprovados
│                       │               │           └─ Não: NÓ 2430 | loan_amount (valor solicitado em USD) <= 1160000.0? | bootstrap: 9924 recusados, 125024 aprovados
│                       │               │               ├─ Sim: NÓ 2431 | lei = '5493000WAJ1FMTFL5D75'? | bootstrap: 9474 recusados, 121473 aprovados
│                       │               │               │   ├─ Sim: FOLHA 2457 | previsão: RECUSADO (0) | bootstrap: 20 recusados, 1 aprovados
│                       │               │               │   └─ Não: NÓ 2432 | lei = '5493003GQDUH26DNNH17'? | bootstrap: 9454 recusados, 121472 aprovados
│                       │               │               │       ├─ Sim: NÓ 2448 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 137 recusados, 1136 aprovados
│                       │               │               │       │   ├─ Sim: FOLHA 2456 | previsão: APROVADO (1) | bootstrap: 26 recusados, 94 aprovados
│                       │               │               │       │   └─ Não: NÓ 2449 | property_value (valor do imóvel em USD) <= 220000.0? | bootstrap: 111 recusados, 1042 aprovados
│                       │               │               │       │       ├─ Sim: NÓ 2450 | property_value (valor do imóvel em USD) <= 200000.0? | bootstrap: 21 recusados, 51 aprovados
│                       │               │               │       │       │   ├─ Sim: FOLHA 2451 | previsão: APROVADO (1) | bootstrap: 11 recusados, 36 aprovados
│                       │               │               │       │       │   └─ Não: FOLHA 2452 | previsão: APROVADO (1) | bootstrap: 10 recusados, 15 aprovados
│                       │               │               │       │       └─ Não: NÓ 2453 | debt_to_income_ratio = '49'? | bootstrap: 90 recusados, 991 aprovados
│                       │               │               │       │           ├─ Sim: FOLHA 2455 | previsão: APROVADO (1) | bootstrap: 1 recusados, 33 aprovados
│                       │               │               │       │           └─ Não: FOLHA 2454 | previsão: APROVADO (1) | bootstrap: 89 recusados, 958 aprovados
│                       │               │               │       └─ Não: NÓ 2433 | lei = '549300FGXN1K3HLB1R50'? | bootstrap: 9317 recusados, 120336 aprovados
│                       │               │               │           ├─ Sim: NÓ 2441 | debt_to_income_ratio = '46'? | bootstrap: 737 recusados, 5741 aprovados
│                       │               │               │           │   ├─ Sim: NÓ 2445 | loan_type = '2'? | bootstrap: 27 recusados, 255 aprovados
│                       │               │               │           │   │   ├─ Sim: FOLHA 2447 | previsão: APROVADO (1) | bootstrap: 5 recusados, 61 aprovados
│                       │               │               │           │   │   └─ Não: FOLHA 2446 | previsão: APROVADO (1) | bootstrap: 22 recusados, 194 aprovados
│                       │               │               │           │   └─ Não: NÓ 2442 | loan_type = '1'? | bootstrap: 710 recusados, 5486 aprovados
│                       │               │               │           │       ├─ Sim: FOLHA 2444 | previsão: APROVADO (1) | bootstrap: 491 recusados, 4418 aprovados
│                       │               │               │           │       └─ Não: FOLHA 2443 | previsão: APROVADO (1) | bootstrap: 219 recusados, 1068 aprovados
│                       │               │               │           └─ Não: NÓ 2434 | lei = '549300MZ8VZJOVC63092'? | bootstrap: 8580 recusados, 114595 aprovados
│                       │               │               │               ├─ Sim: NÓ 2438 | loan_amount (valor solicitado em USD) <= 45000.0? | bootstrap: 140 recusados, 613 aprovados
│                       │               │               │               │   ├─ Sim: FOLHA 2439 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
│                       │               │               │               │   └─ Não: FOLHA 2440 | previsão: APROVADO (1) | bootstrap: 140 recusados, 595 aprovados
│                       │               │               │               └─ Não: NÓ 2435 | lei = '337KMNHEWWWR6B7Q7W10'? | bootstrap: 8440 recusados, 113982 aprovados
│                       │               │               │                   ├─ Sim: FOLHA 2437 | previsão: APROVADO (1) | bootstrap: 6 recusados, 18 aprovados
│                       │               │               │                   └─ Não: FOLHA 2436 | previsão: APROVADO (1) | bootstrap: 8434 recusados, 113964 aprovados
│                       │               │               └─ Não: NÓ 2458 | lei = '549300FK3AFCFVAPH234'? | bootstrap: 450 recusados, 3551 aprovados
│                       │               │                   ├─ Sim: FOLHA 2468 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
│                       │               │                   └─ Não: NÓ 2459 | lei = '5493008NWHQT1R22C024'? | bootstrap: 450 recusados, 3541 aprovados
│                       │               │                       ├─ Sim: FOLHA 2467 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 3 aprovados
│                       │               │                       └─ Não: NÓ 2460 | lei = '5493003GQDUH26DNNH17'? | bootstrap: 444 recusados, 3538 aprovados
│                       │               │                           ├─ Sim: FOLHA 2466 | previsão: APROVADO (1) | bootstrap: 3 recusados, 23 aprovados
│                       │               │                           └─ Não: NÓ 2461 | lei = '549300R77W0P105C3S50'? | bootstrap: 441 recusados, 3515 aprovados
│                       │               │                               ├─ Sim: FOLHA 2465 | previsão: APROVADO (1) | bootstrap: 12 recusados, 17 aprovados
│                       │               │                               └─ Não: NÓ 2462 | debt_to_income_ratio = '45'? | bootstrap: 429 recusados, 3498 aprovados
│                       │               │                                   ├─ Sim: FOLHA 2464 | previsão: APROVADO (1) | bootstrap: 13 recusados, 67 aprovados
│                       │               │                                   └─ Não: FOLHA 2463 | previsão: APROVADO (1) | bootstrap: 416 recusados, 3431 aprovados
│                       │               └─ Não: NÓ 2366 | construction_method = '2'? | bootstrap: 7897 recusados, 50743 aprovados
│                       │                   ├─ Sim: NÓ 2424 | property_value (valor do imóvel em USD) <= 310000.0? | bootstrap: 122 recusados, 361 aprovados
│                       │                   │   ├─ Sim: FOLHA 2425 | previsão: APROVADO (1) | bootstrap: 113 recusados, 277 aprovados
│                       │                   │   └─ Não: FOLHA 2426 | previsão: APROVADO (1) | bootstrap: 9 recusados, 84 aprovados
│                       │                   └─ Não: NÓ 2367 | lei = '549300R77W0P105C3S50'? | bootstrap: 7775 recusados, 50382 aprovados
│                       │                       ├─ Sim: NÓ 2421 | derived_race = 'Black or African American'? | bootstrap: 11 recusados, 18 aprovados
│                       │                       │   ├─ Sim: FOLHA 2423 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 5 aprovados
│                       │                       │   └─ Não: FOLHA 2422 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
│                       │                       └─ Não: NÓ 2368 | lei = '5493002B2WMHY23GFK92'? | bootstrap: 7764 recusados, 50364 aprovados
│                       │                           ├─ Sim: FOLHA 2420 | previsão: APROVADO (1) | bootstrap: 26 recusados, 52 aprovados
│                       │                           └─ Não: NÓ 2369 | lei = '549300H3IZO24NSOO931'? | bootstrap: 7738 recusados, 50312 aprovados
│                       │                               ├─ Sim: NÓ 2391 | loan_amount (valor solicitado em USD) <= 430000.0? | bootstrap: 328 recusados, 1249 aprovados
│                       │                               │   ├─ Sim: NÓ 2392 | debt_to_income_ratio = '45'? | bootstrap: 247 recusados, 758 aprovados
│                       │                               │   │   ├─ Sim: NÓ 2400 | loan_amount (valor solicitado em USD) <= 290000.0? | bootstrap: 5 recusados, 40 aprovados
│                       │                               │   │   │   ├─ Sim: FOLHA 2401 | previsão: APROVADO (1) | bootstrap: 4 recusados, 11 aprovados
│                       │                               │   │   │   └─ Não: NÓ 2402 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 125.5? | bootstrap: 1 recusados, 29 aprovados
│                       │                               │   │   │       ├─ Sim: FOLHA 2403 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
│                       │                               │   │   │       └─ Não: FOLHA 2404 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
│                       │                               │   │   └─ Não: NÓ 2393 | debt_to_income_ratio = '42'? | bootstrap: 242 recusados, 718 aprovados
│                       │                               │   │       ├─ Sim: FOLHA 2399 | previsão: APROVADO (1) | bootstrap: 7 recusados, 17 aprovados
│                       │                               │   │       └─ Não: NÓ 2394 | debt_to_income_ratio = '38'? | bootstrap: 235 recusados, 701 aprovados
│                       │                               │   │           ├─ Sim: FOLHA 2398 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
│                       │                               │   │           └─ Não: NÓ 2395 | debt_to_income_ratio = '37'? | bootstrap: 235 recusados, 684 aprovados
│                       │                               │   │               ├─ Sim: FOLHA 2397 | previsão: APROVADO (1) | bootstrap: 1 recusados, 25 aprovados
│                       │                               │   │               └─ Não: FOLHA 2396 | previsão: APROVADO (1) | bootstrap: 234 recusados, 659 aprovados
│                       │                               │   └─ Não: NÓ 2405 | debt_to_income_ratio = '48'? | bootstrap: 81 recusados, 491 aprovados
│                       │                               │       ├─ Sim: NÓ 2417 | loan_amount (valor solicitado em USD) <= 600000.0? | bootstrap: 1 recusados, 25 aprovados
│                       │                               │       │   ├─ Sim: FOLHA 2418 | previsão: APROVADO (1) | bootstrap: 0 recusados, 21 aprovados
│                       │                               │       │   └─ Não: FOLHA 2419 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
│                       │                               │       └─ Não: NÓ 2406 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 80 recusados, 466 aprovados
│                       │                               │           ├─ Sim: NÓ 2412 | debt_to_income_ratio = '49'? | bootstrap: 13 recusados, 64 aprovados
│                       │                               │           │   ├─ Sim: FOLHA 2416 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
│                       │                               │           │   └─ Não: NÓ 2413 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 103.5? | bootstrap: 13 recusados, 55 aprovados
│                       │                               │           │       ├─ Sim: FOLHA 2414 | previsão: APROVADO (1) | bootstrap: 1 recusados, 24 aprovados
│                       │                               │           │       └─ Não: FOLHA 2415 | previsão: APROVADO (1) | bootstrap: 12 recusados, 31 aprovados
│                       │                               │           └─ Não: NÓ 2407 | debt_to_income_ratio = '41'? | bootstrap: 67 recusados, 402 aprovados
│                       │                               │               ├─ Sim: FOLHA 2411 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
│                       │                               │               └─ Não: NÓ 2408 | debt_to_income_ratio = '45'? | bootstrap: 67 recusados, 383 aprovados
│                       │                               │                   ├─ Sim: FOLHA 2410 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
│                       │                               │                   └─ Não: FOLHA 2409 | previsão: APROVADO (1) | bootstrap: 67 recusados, 355 aprovados
│                       │                               └─ Não: NÓ 2370 | lei = '254900VELJPWWT51WN03'? | bootstrap: 7410 recusados, 49063 aprovados
│                       │                                   ├─ Sim: FOLHA 2390 | previsão: APROVADO (1) | bootstrap: 0 recusados, 156 aprovados
│                       │                                   └─ Não: NÓ 2371 | lei = '549300JYXTZDSPJEPI44'? | bootstrap: 7410 recusados, 48907 aprovados
│                       │                                       ├─ Sim: NÓ 2383 | debt_to_income_ratio = '42'? | bootstrap: 80 recusados, 191 aprovados
│                       │                                       │   ├─ Sim: FOLHA 2389 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
│                       │                                       │   └─ Não: NÓ 2384 | debt_to_income_ratio = '48'? | bootstrap: 78 recusados, 182 aprovados
│                       │                                       │       ├─ Sim: FOLHA 2388 | previsão: APROVADO (1) | bootstrap: 6 recusados, 8 aprovados
│                       │                                       │       └─ Não: NÓ 2385 | debt_to_income_ratio = '40'? | bootstrap: 72 recusados, 174 aprovados
│                       │                                       │           ├─ Sim: FOLHA 2387 | previsão: APROVADO (1) | bootstrap: 1 recusados, 12 aprovados
│                       │                                       │           └─ Não: FOLHA 2386 | previsão: APROVADO (1) | bootstrap: 71 recusados, 162 aprovados
│                       │                                       └─ Não: NÓ 2372 | debt_to_income_ratio = '>60%'? | bootstrap: 7330 recusados, 48716 aprovados
│                       │                                           ├─ Sim: NÓ 2378 | property_value (valor do imóvel em USD) <= 590000.0? | bootstrap: 2073 recusados, 322 aprovados
│                       │                                           │   ├─ Sim: FOLHA 2379 | previsão: RECUSADO (0) | bootstrap: 1527 recusados, 185 aprovados
│                       │                                           │   └─ Não: NÓ 2380 | total_units = '1'? | bootstrap: 546 recusados, 137 aprovados
│                       │                                           │       ├─ Sim: FOLHA 2382 | previsão: RECUSADO (0) | bootstrap: 510 recusados, 133 aprovados
│                       │                                           │       └─ Não: FOLHA 2381 | previsão: RECUSADO (0) | bootstrap: 36 recusados, 4 aprovados
│                       │                                           └─ Não: NÓ 2373 | debt_to_income_ratio = '41'? | bootstrap: 5257 recusados, 48394 aprovados
│                       │                                               ├─ Sim: FOLHA 2377 | previsão: APROVADO (1) | bootstrap: 168 recusados, 2164 aprovados
│                       │                                               └─ Não: NÓ 2374 | lei = '549300EM8ID8J7F8OM55'? | bootstrap: 5089 recusados, 46230 aprovados
│                       │                                                   ├─ Sim: FOLHA 2376 | previsão: APROVADO (1) | bootstrap: 2 recusados, 156 aprovados
│                       │                                                   └─ Não: FOLHA 2375 | previsão: APROVADO (1) | bootstrap: 5087 recusados, 46074 aprovados
│                       └─ Não: NÓ 2607 | property_value (valor do imóvel em USD) <= 380000.0? | bootstrap: 571 recusados, 2547 aprovados
│                           ├─ Sim: FOLHA 2608 | previsão: APROVADO (1) | bootstrap: 205 recusados, 489 aprovados
│                           └─ Não: NÓ 2609 | lei = 'SS1TRMSN6BRNMOREEV51'? | bootstrap: 366 recusados, 2058 aprovados
│                               ├─ Sim: NÓ 2617 | income (renda anual declarada em milhares de USD) <= 148.5? | bootstrap: 19 recusados, 29 aprovados
│                               │   ├─ Sim: FOLHA 2618 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 4 aprovados
│                               │   └─ Não: NÓ 2619 | loan_term (prazo em meses) <= 370.5? | bootstrap: 9 recusados, 25 aprovados
│                               │       ├─ Sim: FOLHA 2620 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
│                               │       └─ Não: NÓ 2621 | derived_race = 'White'? | bootstrap: 8 recusados, 17 aprovados
│                               │           ├─ Sim: FOLHA 2623 | previsão: APROVADO (1) | bootstrap: 5 recusados, 14 aprovados
│                               │           └─ Não: FOLHA 2622 | previsão: RECUSADO (0) | bootstrap: 3 recusados, 3 aprovados
│                               └─ Não: NÓ 2610 | lei = '549300E2UX99HKDBR481'? | bootstrap: 347 recusados, 2029 aprovados
│                                   ├─ Sim: FOLHA 2616 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
│                                   └─ Não: NÓ 2611 | debt_to_income_ratio = '44'? | bootstrap: 347 recusados, 2020 aprovados
│                                       ├─ Sim: NÓ 2613 | lien_status = '2'? | bootstrap: 14 recusados, 112 aprovados
│                                       │   ├─ Sim: FOLHA 2615 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
│                                       │   └─ Não: FOLHA 2614 | previsão: APROVADO (1) | bootstrap: 14 recusados, 101 aprovados
│                                       └─ Não: FOLHA 2612 | previsão: APROVADO (1) | bootstrap: 333 recusados, 1908 aprovados
└─ Não: NÓ 1 | lei = '549300MGPZBLQDIL7538'? | bootstrap: 167817 recusados, 1302648 aprovados
    ├─ Sim: NÓ 1629 | debt_to_income_ratio = '50%-60%'? | bootstrap: 1336 recusados, 26578 aprovados
    │   ├─ Sim: NÓ 2105 | total_units = '2'? | bootstrap: 278 recusados, 2548 aprovados
    │   │   ├─ Sim: NÓ 2125 | derived_race = 'Black or African American'? | bootstrap: 14 recusados, 36 aprovados
    │   │   │   ├─ Sim: FOLHA 2127 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 8 aprovados
    │   │   │   └─ Não: FOLHA 2126 | previsão: APROVADO (1) | bootstrap: 5 recusados, 28 aprovados
    │   │   └─ Não: NÓ 2106 | derived_ethnicity = 'Joint'? | bootstrap: 264 recusados, 2512 aprovados
    │   │       ├─ Sim: NÓ 2120 | applicant_age = '35-44'? | bootstrap: 12 recusados, 118 aprovados
    │   │       │   ├─ Sim: FOLHA 2124 | previsão: APROVADO (1) | bootstrap: 6 recusados, 48 aprovados
    │   │       │   └─ Não: NÓ 2121 | derived_race = 'Joint'? | bootstrap: 6 recusados, 70 aprovados
    │   │       │       ├─ Sim: FOLHA 2123 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
    │   │       │       └─ Não: FOLHA 2122 | previsão: APROVADO (1) | bootstrap: 6 recusados, 56 aprovados
    │   │       └─ Não: NÓ 2107 | applicant_age = '>74'? | bootstrap: 252 recusados, 2394 aprovados
    │   │           ├─ Sim: NÓ 2117 | loan_amount (valor solicitado em USD) <= 200000.0? | bootstrap: 4 recusados, 45 aprovados
    │   │           │   ├─ Sim: FOLHA 2118 | previsão: APROVADO (1) | bootstrap: 4 recusados, 9 aprovados
    │   │           │   └─ Não: FOLHA 2119 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
    │   │           └─ Não: NÓ 2108 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 31.0? | bootstrap: 248 recusados, 2349 aprovados
    │   │               ├─ Sim: FOLHA 2109 | previsão: APROVADO (1) | bootstrap: 4 recusados, 7 aprovados
    │   │               └─ Não: NÓ 2110 | applicant_age = '65-74'? | bootstrap: 244 recusados, 2342 aprovados
    │   │                   ├─ Sim: NÓ 2112 | derived_race = 'White'? | bootstrap: 9 recusados, 94 aprovados
    │   │                   │   ├─ Sim: NÓ 2114 | lien_status = '2'? | bootstrap: 8 recusados, 64 aprovados
    │   │                   │   │   ├─ Sim: FOLHA 2116 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
    │   │                   │   │   └─ Não: FOLHA 2115 | previsão: APROVADO (1) | bootstrap: 7 recusados, 57 aprovados
    │   │                   │   └─ Não: FOLHA 2113 | previsão: APROVADO (1) | bootstrap: 1 recusados, 30 aprovados
    │   │                   └─ Não: FOLHA 2111 | previsão: APROVADO (1) | bootstrap: 235 recusados, 2248 aprovados
    │   └─ Não: NÓ 1630 | loan_amount (valor solicitado em USD) <= 180000.0? | bootstrap: 1058 recusados, 24030 aprovados
    │       ├─ Sim: NÓ 1631 | loan_type = '1'? | bootstrap: 474 recusados, 5385 aprovados
    │       │   ├─ Sim: NÓ 1707 | total_units = '1'? | bootstrap: 324 recusados, 4239 aprovados
    │       │   │   ├─ Sim: NÓ 1711 | debt_to_income_ratio = '39'? | bootstrap: 322 recusados, 4184 aprovados
    │       │   │   │   ├─ Sim: FOLHA 1865 | previsão: APROVADO (1) | bootstrap: 2 recusados, 168 aprovados
    │       │   │   │   └─ Não: NÓ 1712 | applicant_age = '25-34'? | bootstrap: 320 recusados, 4016 aprovados
    │       │   │   │       ├─ Sim: NÓ 1824 | derived_race = 'Joint'? | bootstrap: 120 recusados, 1568 aprovados
    │       │   │   │       │   ├─ Sim: FOLHA 1864 | previsão: APROVADO (1) | bootstrap: 2 recusados, 49 aprovados
    │       │   │   │       │   └─ Não: NÓ 1825 | income (renda anual declarada em milhares de USD) <= 28.5? | bootstrap: 118 recusados, 1519 aprovados
    │       │   │   │       │       ├─ Sim: FOLHA 1826 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 4 aprovados
    │       │   │   │       │       └─ Não: NÓ 1827 | derived_race = 'White'? | bootstrap: 111 recusados, 1515 aprovados
    │       │   │   │       │           ├─ Sim: NÓ 1833 | debt_to_income_ratio = '47'? | bootstrap: 84 recusados, 1335 aprovados
    │       │   │   │       │           │   ├─ Sim: NÓ 1859 | property_value (valor do imóvel em USD) <= 370000.0? | bootstrap: 7 recusados, 39 aprovados
    │       │   │   │       │           │   │   ├─ Sim: NÓ 1860 | loan_amount (valor solicitado em USD) <= 105000.0? | bootstrap: 1 recusados, 33 aprovados
    │       │   │   │       │           │   │   │   ├─ Sim: FOLHA 1861 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
    │       │   │   │       │           │   │   │   └─ Não: FOLHA 1862 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
    │       │   │   │       │           │   │   └─ Não: FOLHA 1863 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 6 aprovados
    │       │   │   │       │           │   └─ Não: NÓ 1834 | property_value (valor do imóvel em USD) <= 500000.0? | bootstrap: 77 recusados, 1296 aprovados
    │       │   │   │       │           │       ├─ Sim: NÓ 1835 | construction_method = '1'? | bootstrap: 70 recusados, 1274 aprovados
    │       │   │   │       │           │       │   ├─ Sim: NÓ 1837 | property_value (valor do imóvel em USD) <= 200000.0? | bootstrap: 67 recusados, 1221 aprovados
    │       │   │   │       │           │       │   │   ├─ Sim: NÓ 1838 | loan_term (prazo em meses) <= 150.0? | bootstrap: 41 recusados, 591 aprovados
    │       │   │   │       │           │       │   │   │   ├─ Sim: FOLHA 1839 | previsão: APROVADO (1) | bootstrap: 7 recusados, 112 aprovados
    │       │   │   │       │           │       │   │   │   └─ Não: FOLHA 1840 | previsão: APROVADO (1) | bootstrap: 34 recusados, 479 aprovados
    │       │   │   │       │           │       │   │   └─ Não: NÓ 1841 | debt_to_income_ratio = '44'? | bootstrap: 26 recusados, 630 aprovados
    │       │   │   │       │           │       │   │       ├─ Sim: NÓ 1855 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 1 recusados, 40 aprovados
    │       │   │   │       │           │       │   │       │   ├─ Sim: FOLHA 1857 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
    │       │   │   │       │           │       │   │       │   └─ Não: FOLHA 1856 | previsão: APROVADO (1) | bootstrap: 0 recusados, 27 aprovados
    │       │   │   │       │           │       │   │       └─ Não: NÓ 1842 | income (renda anual declarada em milhares de USD) <= 106.5? | bootstrap: 25 recusados, 590 aprovados
    │       │   │   │       │           │       │   │           ├─ Sim: NÓ 1843 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 76.5? | bootstrap: 12 recusados, 461 aprovados
    │       │   │   │       │           │       │   │           │   ├─ Sim: FOLHA 1844 | previsão: APROVADO (1) | bootstrap: 0 recusados, 104 aprovados
    │       │   │   │       │           │       │   │           │   └─ Não: NÓ 1845 | debt_to_income_ratio = '<20%'? | bootstrap: 12 recusados, 357 aprovados
    │       │   │   │       │           │       │   │           │       ├─ Sim: NÓ 1849 | loan_term (prazo em meses) <= 300.0? | bootstrap: 4 recusados, 28 aprovados
    │       │   │   │       │           │       │   │           │       │   ├─ Sim: FOLHA 1850 | previsão: APROVADO (1) | bootstrap: 1 recusados, 17 aprovados
    │       │   │   │       │           │       │   │           │       │   └─ Não: FOLHA 1851 | previsão: APROVADO (1) | bootstrap: 3 recusados, 11 aprovados
    │       │   │   │       │           │       │   │           │       └─ Não: NÓ 1846 | income (renda anual declarada em milhares de USD) <= 56.5? | bootstrap: 8 recusados, 329 aprovados
    │       │   │   │       │           │       │   │           │           ├─ Sim: FOLHA 1847 | previsão: APROVADO (1) | bootstrap: 3 recusados, 45 aprovados
    │       │   │   │       │           │       │   │           │           └─ Não: FOLHA 1848 | previsão: APROVADO (1) | bootstrap: 5 recusados, 284 aprovados
    │       │   │   │       │           │       │   │           └─ Não: NÓ 1852 | debt_to_income_ratio = '42'? | bootstrap: 13 recusados, 129 aprovados
    │       │   │   │       │           │       │   │               ├─ Sim: FOLHA 1854 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │       │   │   │       │           │       │   │               └─ Não: FOLHA 1853 | previsão: APROVADO (1) | bootstrap: 13 recusados, 119 aprovados
    │       │   │   │       │           │       │   └─ Não: FOLHA 1836 | previsão: APROVADO (1) | bootstrap: 3 recusados, 53 aprovados
    │       │   │   │       │           │       └─ Não: FOLHA 1858 | previsão: APROVADO (1) | bootstrap: 7 recusados, 22 aprovados
    │       │   │   │       │           └─ Não: NÓ 1828 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 27 recusados, 180 aprovados
    │       │   │   │       │               ├─ Sim: NÓ 1830 | derived_race = 'Black or African American'? | bootstrap: 22 recusados, 162 aprovados
    │       │   │   │       │               │   ├─ Sim: FOLHA 1832 | previsão: APROVADO (1) | bootstrap: 22 recusados, 108 aprovados
    │       │   │   │       │               │   └─ Não: FOLHA 1831 | previsão: APROVADO (1) | bootstrap: 0 recusados, 54 aprovados
    │       │   │   │       │               └─ Não: FOLHA 1829 | previsão: APROVADO (1) | bootstrap: 5 recusados, 18 aprovados
    │       │   │   │       └─ Não: NÓ 1713 | debt_to_income_ratio = '37'? | bootstrap: 200 recusados, 2448 aprovados
    │       │   │   │           ├─ Sim: NÓ 1813 | applicant_age = '35-44'? | bootstrap: 17 recusados, 77 aprovados
    │       │   │   │           │   ├─ Sim: NÓ 1819 | lien_status = '1'? | bootstrap: 10 recusados, 23 aprovados
    │       │   │   │           │   │   ├─ Sim: NÓ 1821 | property_value (valor do imóvel em USD) <= 200000.0? | bootstrap: 7 recusados, 11 aprovados
    │       │   │   │           │   │   │   ├─ Sim: FOLHA 1822 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
    │       │   │   │           │   │   │   └─ Não: FOLHA 1823 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
    │       │   │   │           │   │   └─ Não: FOLHA 1820 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
    │       │   │   │           │   └─ Não: NÓ 1814 | loan_amount (valor solicitado em USD) <= 20000.0? | bootstrap: 7 recusados, 54 aprovados
    │       │   │   │           │       ├─ Sim: FOLHA 1815 | previsão: APROVADO (1) | bootstrap: 5 recusados, 15 aprovados
    │       │   │   │           │       └─ Não: NÓ 1816 | applicant_age = '<25'? | bootstrap: 2 recusados, 39 aprovados
    │       │   │   │           │           ├─ Sim: FOLHA 1818 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
    │       │   │   │           │           └─ Não: FOLHA 1817 | previsão: APROVADO (1) | bootstrap: 2 recusados, 25 aprovados
    │       │   │   │           └─ Não: NÓ 1714 | derived_ethnicity = 'Joint'? | bootstrap: 183 recusados, 2371 aprovados
    │       │   │   │               ├─ Sim: NÓ 1810 | derived_race = 'White'? | bootstrap: 8 recusados, 61 aprovados
    │       │   │   │               │   ├─ Sim: FOLHA 1812 | previsão: APROVADO (1) | bootstrap: 3 recusados, 57 aprovados
    │       │   │   │               │   └─ Não: FOLHA 1811 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
    │       │   │   │               └─ Não: NÓ 1715 | applicant_age = '35-44'? | bootstrap: 175 recusados, 2310 aprovados
    │       │   │   │                   ├─ Sim: NÓ 1779 | debt_to_income_ratio = '41'? | bootstrap: 71 recusados, 781 aprovados
    │       │   │   │                   │   ├─ Sim: NÓ 1805 | income (renda anual declarada em milhares de USD) <= 82.0? | bootstrap: 9 recusados, 42 aprovados
    │       │   │   │                   │   │   ├─ Sim: NÓ 1806 | derived_race = 'White'? | bootstrap: 9 recusados, 23 aprovados
    │       │   │   │                   │   │   │   ├─ Sim: FOLHA 1808 | previsão: APROVADO (1) | bootstrap: 4 recusados, 21 aprovados
    │       │   │   │                   │   │   │   └─ Não: FOLHA 1807 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 2 aprovados
    │       │   │   │                   │   │   └─ Não: FOLHA 1809 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
    │       │   │   │                   │   └─ Não: NÓ 1780 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 37.5? | bootstrap: 62 recusados, 739 aprovados
    │       │   │   │                   │       ├─ Sim: FOLHA 1781 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
    │       │   │   │                   │       └─ Não: NÓ 1782 | derived_race = 'Asian'? | bootstrap: 59 recusados, 730 aprovados
    │       │   │   │                   │           ├─ Sim: FOLHA 1804 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
    │       │   │   │                   │           └─ Não: NÓ 1783 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 59 recusados, 694 aprovados
    │       │   │   │                   │               ├─ Sim: FOLHA 1803 | previsão: APROVADO (1) | bootstrap: 7 recusados, 128 aprovados
    │       │   │   │                   │               └─ Não: NÓ 1784 | debt_to_income_ratio = '43'? | bootstrap: 52 recusados, 566 aprovados
    │       │   │   │                   │                   ├─ Sim: NÓ 1796 | loan_amount (valor solicitado em USD) <= 140000.0? | bootstrap: 6 recusados, 34 aprovados
    │       │   │   │                   │                   │   ├─ Sim: NÓ 1797 | loan_term (prazo em meses) <= 270.0? | bootstrap: 6 recusados, 22 aprovados
    │       │   │   │                   │                   │   │   ├─ Sim: NÓ 1798 | income (renda anual declarada em milhares de USD) <= 67.0? | bootstrap: 2 recusados, 14 aprovados
    │       │   │   │                   │                   │   │   │   ├─ Sim: FOLHA 1799 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
    │       │   │   │                   │                   │   │   │   └─ Não: FOLHA 1800 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │       │   │   │                   │                   │   │   └─ Não: FOLHA 1801 | previsão: APROVADO (1) | bootstrap: 4 recusados, 8 aprovados
    │       │   │   │                   │                   │   └─ Não: FOLHA 1802 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
    │       │   │   │                   │                   └─ Não: NÓ 1785 | debt_to_income_ratio = '38'? | bootstrap: 46 recusados, 532 aprovados
    │       │   │   │                   │                       ├─ Sim: FOLHA 1795 | previsão: APROVADO (1) | bootstrap: 2 recusados, 34 aprovados
    │       │   │   │                   │                       └─ Não: NÓ 1786 | debt_to_income_ratio = '46'? | bootstrap: 44 recusados, 498 aprovados
    │       │   │   │                   │                           ├─ Sim: NÓ 1792 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 2 recusados, 28 aprovados
    │       │   │   │                   │                           │   ├─ Sim: FOLHA 1794 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
    │       │   │   │                   │                           │   └─ Não: FOLHA 1793 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
    │       │   │   │                   │                           └─ Não: NÓ 1787 | debt_to_income_ratio = '36'? | bootstrap: 42 recusados, 470 aprovados
    │       │   │   │                   │                               ├─ Sim: FOLHA 1791 | previsão: APROVADO (1) | bootstrap: 2 recusados, 23 aprovados
    │       │   │   │                   │                               └─ Não: NÓ 1788 | debt_to_income_ratio = '48'? | bootstrap: 40 recusados, 447 aprovados
    │       │   │   │                   │                                   ├─ Sim: FOLHA 1790 | previsão: APROVADO (1) | bootstrap: 4 recusados, 22 aprovados
    │       │   │   │                   │                                   └─ Não: FOLHA 1789 | previsão: APROVADO (1) | bootstrap: 36 recusados, 425 aprovados
    │       │   │   │                   └─ Não: NÓ 1716 | debt_to_income_ratio = '44'? | bootstrap: 104 recusados, 1529 aprovados
    │       │   │   │                       ├─ Sim: NÓ 1770 | income (renda anual declarada em milhares de USD) <= 36.0? | bootstrap: 3 recusados, 74 aprovados
    │       │   │   │                       │   ├─ Sim: FOLHA 1771 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
    │       │   │   │                       │   └─ Não: NÓ 1772 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 1 recusados, 65 aprovados
    │       │   │   │                       │       ├─ Sim: FOLHA 1778 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
    │       │   │   │                       │       └─ Não: NÓ 1773 | applicant_age = '<25'? | bootstrap: 1 recusados, 51 aprovados
    │       │   │   │                       │           ├─ Sim: FOLHA 1777 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
    │       │   │   │                       │           └─ Não: NÓ 1774 | loan_term (prazo em meses) <= 300.0? | bootstrap: 1 recusados, 25 aprovados
    │       │   │   │                       │               ├─ Sim: FOLHA 1775 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
    │       │   │   │                       │               └─ Não: FOLHA 1776 | previsão: APROVADO (1) | bootstrap: 1 recusados, 18 aprovados
    │       │   │   │                       └─ Não: NÓ 1717 | debt_to_income_ratio = '45'? | bootstrap: 101 recusados, 1455 aprovados
    │       │   │   │                           ├─ Sim: NÓ 1765 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 2 recusados, 62 aprovados
    │       │   │   │                           │   ├─ Sim: NÓ 1767 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 49.0? | bootstrap: 2 recusados, 55 aprovados
    │       │   │   │                           │   │   ├─ Sim: FOLHA 1768 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
    │       │   │   │                           │   │   └─ Não: FOLHA 1769 | previsão: APROVADO (1) | bootstrap: 1 recusados, 48 aprovados
    │       │   │   │                           │   └─ Não: FOLHA 1766 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
    │       │   │   │                           └─ Não: NÓ 1718 | applicant_age = '55-64'? | bootstrap: 99 recusados, 1393 aprovados
    │       │   │   │                               ├─ Sim: NÓ 1750 | construction_method = '1'? | bootstrap: 36 recusados, 422 aprovados
    │       │   │   │                               │   ├─ Sim: NÓ 1752 | debt_to_income_ratio = '46'? | bootstrap: 36 recusados, 400 aprovados
    │       │   │   │                               │   │   ├─ Sim: NÓ 1762 | income (renda anual declarada em milhares de USD) <= 64.5? | bootstrap: 6 recusados, 17 aprovados
    │       │   │   │                               │   │   │   ├─ Sim: FOLHA 1763 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
    │       │   │   │                               │   │   │   └─ Não: FOLHA 1764 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
    │       │   │   │                               │   │   └─ Não: NÓ 1753 | loan_amount (valor solicitado em USD) <= 100000.0? | bootstrap: 30 recusados, 383 aprovados
    │       │   │   │                               │   │       ├─ Sim: NÓ 1754 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 20 recusados, 146 aprovados
    │       │   │   │                               │   │       │   ├─ Sim: FOLHA 1760 | previsão: APROVADO (1) | bootstrap: 18 recusados, 114 aprovados
    │       │   │   │                               │   │       │   └─ Não: NÓ 1755 | debt_to_income_ratio = '48'? | bootstrap: 2 recusados, 32 aprovados
    │       │   │   │                               │   │       │       ├─ Sim: FOLHA 1759 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │       │   │   │                               │   │       │       └─ Não: NÓ 1756 | loan_term (prazo em meses) <= 150.0? | bootstrap: 2 recusados, 22 aprovados
    │       │   │   │                               │   │       │           ├─ Sim: FOLHA 1757 | previsão: APROVADO (1) | bootstrap: 2 recusados, 14 aprovados
    │       │   │   │                               │   │       │           └─ Não: FOLHA 1758 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │       │   │   │                               │   │       └─ Não: FOLHA 1761 | previsão: APROVADO (1) | bootstrap: 10 recusados, 237 aprovados
    │       │   │   │                               │   └─ Não: FOLHA 1751 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
    │       │   │   │                               └─ Não: NÓ 1719 | debt_to_income_ratio = '42'? | bootstrap: 63 recusados, 971 aprovados
    │       │   │   │                                   ├─ Sim: FOLHA 1749 | previsão: APROVADO (1) | bootstrap: 0 recusados, 62 aprovados
    │       │   │   │                                   └─ Não: NÓ 1720 | lien_status = '2'? | bootstrap: 63 recusados, 909 aprovados
    │       │   │   │                                       ├─ Sim: NÓ 1738 | applicant_age = '65-74'? | bootstrap: 34 recusados, 378 aprovados
    │       │   │   │                                       │   ├─ Sim: FOLHA 1748 | previsão: APROVADO (1) | bootstrap: 5 recusados, 50 aprovados
    │       │   │   │                                       │   └─ Não: NÓ 1739 | debt_to_income_ratio = '38'? | bootstrap: 29 recusados, 328 aprovados
    │       │   │   │                                       │       ├─ Sim: NÓ 1745 | property_value (valor do imóvel em USD) <= 230000.0? | bootstrap: 1 recusados, 20 aprovados
    │       │   │   │                                       │       │   ├─ Sim: FOLHA 1746 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
    │       │   │   │                                       │       │   └─ Não: FOLHA 1747 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
    │       │   │   │                                       │       └─ Não: NÓ 1740 | debt_to_income_ratio = '46'? | bootstrap: 28 recusados, 308 aprovados
    │       │   │   │                                       │           ├─ Sim: FOLHA 1744 | previsão: APROVADO (1) | bootstrap: 5 recusados, 24 aprovados
    │       │   │   │                                       │           └─ Não: NÓ 1741 | debt_to_income_ratio = '<20%'? | bootstrap: 23 recusados, 284 aprovados
    │       │   │   │                                       │               ├─ Sim: FOLHA 1743 | previsão: APROVADO (1) | bootstrap: 9 recusados, 46 aprovados
    │       │   │   │                                       │               └─ Não: FOLHA 1742 | previsão: APROVADO (1) | bootstrap: 14 recusados, 238 aprovados
    │       │   │   │                                       └─ Não: NÓ 1721 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 115.5? | bootstrap: 29 recusados, 531 aprovados
    │       │   │   │                                           ├─ Sim: NÓ 1722 | derived_race = 'Black or African American'? | bootstrap: 27 recusados, 408 aprovados
    │       │   │   │                                           │   ├─ Sim: FOLHA 1730 | previsão: APROVADO (1) | bootstrap: 2 recusados, 19 aprovados
    │       │   │   │                                           │   └─ Não: NÓ 1723 | debt_to_income_ratio = '47'? | bootstrap: 25 recusados, 389 aprovados
    │       │   │   │                                           │       ├─ Sim: NÓ 1727 | applicant_age = '65-74'? | bootstrap: 5 recusados, 12 aprovados
    │       │   │   │                                           │       │   ├─ Sim: FOLHA 1729 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
    │       │   │   │                                           │       │   └─ Não: FOLHA 1728 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
    │       │   │   │                                           │       └─ Não: NÓ 1724 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 20 recusados, 377 aprovados
    │       │   │   │                                           │           ├─ Sim: FOLHA 1726 | previsão: APROVADO (1) | bootstrap: 15 recusados, 350 aprovados
    │       │   │   │                                           │           └─ Não: FOLHA 1725 | previsão: APROVADO (1) | bootstrap: 5 recusados, 27 aprovados
    │       │   │   │                                           └─ Não: NÓ 1731 | derived_race = 'White'? | bootstrap: 2 recusados, 123 aprovados
    │       │   │   │                                               ├─ Sim: NÓ 1733 | applicant_age = '>74'? | bootstrap: 1 recusados, 115 aprovados
    │       │   │   │                                               │   ├─ Sim: FOLHA 1737 | previsão: APROVADO (1) | bootstrap: 0 recusados, 27 aprovados
    │       │   │   │                                               │   └─ Não: NÓ 1734 | debt_to_income_ratio = '<20%'? | bootstrap: 1 recusados, 88 aprovados
    │       │   │   │                                               │       ├─ Sim: FOLHA 1736 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │       │   │   │                                               │       └─ Não: FOLHA 1735 | previsão: APROVADO (1) | bootstrap: 1 recusados, 78 aprovados
    │       │   │   │                                               └─ Não: FOLHA 1732 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
    │       │   │   └─ Não: NÓ 1708 | applicant_age = '35-44'? | bootstrap: 2 recusados, 55 aprovados
    │       │   │       ├─ Sim: FOLHA 1710 | previsão: APROVADO (1) | bootstrap: 2 recusados, 19 aprovados
    │       │   │       └─ Não: FOLHA 1709 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
    │       │   └─ Não: NÓ 1632 | loan_type = '3'? | bootstrap: 150 recusados, 1146 aprovados
    │       │       ├─ Sim: NÓ 1696 | derived_race = 'Black or African American'? | bootstrap: 10 recusados, 150 aprovados
    │       │       │   ├─ Sim: FOLHA 1706 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
    │       │       │   └─ Não: NÓ 1697 | derived_race = 'White'? | bootstrap: 10 recusados, 132 aprovados
    │       │       │       ├─ Sim: NÓ 1699 | applicant_age = '25-34'? | bootstrap: 7 recusados, 127 aprovados
    │       │       │       │   ├─ Sim: NÓ 1701 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 100.5? | bootstrap: 3 recusados, 26 aprovados
    │       │       │       │   │   ├─ Sim: NÓ 1702 | property_value (valor do imóvel em USD) <= 145000.0? | bootstrap: 3 recusados, 13 aprovados
    │       │       │       │   │   │   ├─ Sim: FOLHA 1703 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
    │       │       │       │   │   │   └─ Não: FOLHA 1704 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
    │       │       │       │   │   └─ Não: FOLHA 1705 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │       │       │       │   └─ Não: FOLHA 1700 | previsão: APROVADO (1) | bootstrap: 4 recusados, 101 aprovados
    │       │       │       └─ Não: FOLHA 1698 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
    │       │       └─ Não: NÓ 1633 | debt_to_income_ratio = '42'? | bootstrap: 140 recusados, 996 aprovados
    │       │           ├─ Sim: FOLHA 1695 | previsão: APROVADO (1) | bootstrap: 7 recusados, 49 aprovados
    │       │           └─ Não: NÓ 1634 | derived_race = 'White'? | bootstrap: 133 recusados, 947 aprovados
    │       │               ├─ Sim: NÓ 1648 | income (renda anual declarada em milhares de USD) <= 30.5? | bootstrap: 92 recusados, 787 aprovados
    │       │               │   ├─ Sim: FOLHA 1649 | previsão: APROVADO (1) | bootstrap: 6 recusados, 8 aprovados
    │       │               │   └─ Não: NÓ 1650 | property_value (valor do imóvel em USD) <= 70000.0? | bootstrap: 86 recusados, 779 aprovados
    │       │               │       ├─ Sim: FOLHA 1651 | previsão: APROVADO (1) | bootstrap: 5 recusados, 8 aprovados
    │       │               │       └─ Não: NÓ 1652 | debt_to_income_ratio = '<20%'? | bootstrap: 81 recusados, 771 aprovados
    │       │               │           ├─ Sim: FOLHA 1694 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
    │       │               │           └─ Não: NÓ 1653 | loan_amount (valor solicitado em USD) <= 140000.0? | bootstrap: 79 recusados, 764 aprovados
    │       │               │               ├─ Sim: NÓ 1654 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 44 recusados, 349 aprovados
    │       │               │               │   ├─ Sim: NÓ 1658 | applicant_age = '<25'? | bootstrap: 2 recusados, 63 aprovados
    │       │               │               │   │   ├─ Sim: FOLHA 1664 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
    │       │               │               │   │   └─ Não: NÓ 1659 | loan_type = '2'? | bootstrap: 2 recusados, 52 aprovados
    │       │               │               │   │       ├─ Sim: NÓ 1661 | property_value (valor do imóvel em USD) <= 90000.0? | bootstrap: 2 recusados, 35 aprovados
    │       │               │               │   │       │   ├─ Sim: FOLHA 1662 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
    │       │               │               │   │       │   └─ Não: FOLHA 1663 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
    │       │               │               │   │       └─ Não: FOLHA 1660 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
    │       │               │               │   └─ Não: NÓ 1655 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 42 recusados, 286 aprovados
    │       │               │               │       ├─ Sim: FOLHA 1657 | previsão: APROVADO (1) | bootstrap: 38 recusados, 256 aprovados
    │       │               │               │       └─ Não: FOLHA 1656 | previsão: APROVADO (1) | bootstrap: 4 recusados, 30 aprovados
    │       │               │               └─ Não: NÓ 1665 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 35 recusados, 415 aprovados
    │       │               │                   ├─ Sim: NÓ 1689 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 63.0? | bootstrap: 2 recusados, 56 aprovados
    │       │               │                   │   ├─ Sim: FOLHA 1690 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
    │       │               │                   │   └─ Não: NÓ 1691 | income (renda anual declarada em milhares de USD) <= 65.5? | bootstrap: 1 recusados, 50 aprovados
    │       │               │                   │       ├─ Sim: FOLHA 1692 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
    │       │               │                   │       └─ Não: FOLHA 1693 | previsão: APROVADO (1) | bootstrap: 0 recusados, 37 aprovados
    │       │               │                   └─ Não: NÓ 1666 | debt_to_income_ratio = '49'? | bootstrap: 33 recusados, 359 aprovados
    │       │               │                       ├─ Sim: FOLHA 1688 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
    │       │               │                       └─ Não: NÓ 1667 | applicant_age = '35-44'? | bootstrap: 33 recusados, 333 aprovados
    │       │               │                           ├─ Sim: NÓ 1681 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 14 recusados, 82 aprovados
    │       │               │                           │   ├─ Sim: FOLHA 1687 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
    │       │               │                           │   └─ Não: NÓ 1682 | construction_method = '1'? | bootstrap: 14 recusados, 56 aprovados
    │       │               │                           │       ├─ Sim: NÓ 1684 | debt_to_income_ratio = '44'? | bootstrap: 14 recusados, 46 aprovados
    │       │               │                           │       │   ├─ Sim: FOLHA 1686 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
    │       │               │                           │       │   └─ Não: FOLHA 1685 | previsão: APROVADO (1) | bootstrap: 12 recusados, 41 aprovados
    │       │               │                           │       └─ Não: FOLHA 1683 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │       │               │                           └─ Não: NÓ 1668 | loan_type = '2'? | bootstrap: 19 recusados, 251 aprovados
    │       │               │                               ├─ Sim: NÓ 1672 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 16 recusados, 226 aprovados
    │       │               │                               │   ├─ Sim: FOLHA 1680 | previsão: APROVADO (1) | bootstrap: 2 recusados, 56 aprovados
    │       │               │                               │   └─ Não: NÓ 1673 | debt_to_income_ratio = '47'? | bootstrap: 14 recusados, 170 aprovados
    │       │               │                               │       ├─ Sim: FOLHA 1679 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
    │       │               │                               │       └─ Não: NÓ 1674 | applicant_age = '55-64'? | bootstrap: 14 recusados, 163 aprovados
    │       │               │                               │           ├─ Sim: FOLHA 1678 | previsão: APROVADO (1) | bootstrap: 4 recusados, 11 aprovados
    │       │               │                               │           └─ Não: NÓ 1675 | loan_amount (valor solicitado em USD) <= 170000.0? | bootstrap: 10 recusados, 152 aprovados
    │       │               │                               │               ├─ Sim: FOLHA 1676 | previsão: APROVADO (1) | bootstrap: 4 recusados, 108 aprovados
    │       │               │                               │               └─ Não: FOLHA 1677 | previsão: APROVADO (1) | bootstrap: 6 recusados, 44 aprovados
    │       │               │                               └─ Não: NÓ 1669 | property_value (valor do imóvel em USD) <= 170000.0? | bootstrap: 3 recusados, 25 aprovados
    │       │               │                                   ├─ Sim: FOLHA 1670 | previsão: APROVADO (1) | bootstrap: 1 recusados, 15 aprovados
    │       │               │                                   └─ Não: FOLHA 1671 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
    │       │               └─ Não: NÓ 1635 | applicant_age = '65-74'? | bootstrap: 41 recusados, 160 aprovados
    │       │                   ├─ Sim: FOLHA 1647 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │       │                   └─ Não: NÓ 1636 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 41 recusados, 150 aprovados
    │       │                       ├─ Sim: FOLHA 1646 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
    │       │                       └─ Não: NÓ 1637 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 41 recusados, 141 aprovados
    │       │                           ├─ Sim: FOLHA 1645 | previsão: APROVADO (1) | bootstrap: 2 recusados, 25 aprovados
    │       │                           └─ Não: NÓ 1638 | debt_to_income_ratio = '41'? | bootstrap: 39 recusados, 116 aprovados
    │       │                               ├─ Sim: FOLHA 1644 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
    │       │                               └─ Não: NÓ 1639 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 38 recusados, 108 aprovados
    │       │                                   ├─ Sim: NÓ 1641 | property_value (valor do imóvel em USD) <= 150000.0? | bootstrap: 6 recusados, 19 aprovados
    │       │                                   │   ├─ Sim: FOLHA 1642 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │       │                                   │   └─ Não: FOLHA 1643 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 6 aprovados
    │       │                                   └─ Não: FOLHA 1640 | previsão: APROVADO (1) | bootstrap: 32 recusados, 89 aprovados
    │       └─ Não: NÓ 1866 | debt_to_income_ratio = '42'? | bootstrap: 584 recusados, 18645 aprovados
    │           ├─ Sim: NÓ 2092 | total_units = '2'? | bootstrap: 26 recusados, 956 aprovados
    │           │   ├─ Sim: FOLHA 2104 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │           │   └─ Não: NÓ 2093 | loan_term (prazo em meses) <= 330.0? | bootstrap: 26 recusados, 943 aprovados
    │           │       ├─ Sim: FOLHA 2094 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
    │           │       └─ Não: NÓ 2095 | derived_race = 'White'? | bootstrap: 26 recusados, 925 aprovados
    │           │           ├─ Sim: FOLHA 2103 | previsão: APROVADO (1) | bootstrap: 13 recusados, 761 aprovados
    │           │           └─ Não: NÓ 2096 | loan_type = '1'? | bootstrap: 13 recusados, 164 aprovados
    │           │               ├─ Sim: NÓ 2098 | applicant_age = '55-64'? | bootstrap: 7 recusados, 102 aprovados
    │           │               │   ├─ Sim: FOLHA 2102 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
    │           │               │   └─ Não: NÓ 2099 | derived_race = 'Asian'? | bootstrap: 6 recusados, 94 aprovados
    │           │               │       ├─ Sim: FOLHA 2101 | previsão: APROVADO (1) | bootstrap: 0 recusados, 41 aprovados
    │           │               │       └─ Não: FOLHA 2100 | previsão: APROVADO (1) | bootstrap: 6 recusados, 53 aprovados
    │           │               └─ Não: FOLHA 2097 | previsão: APROVADO (1) | bootstrap: 6 recusados, 62 aprovados
    │           └─ Não: NÓ 1867 | income (renda anual declarada em milhares de USD) <= 35.5? | bootstrap: 558 recusados, 17689 aprovados
    │               ├─ Sim: FOLHA 1868 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 0 aprovados
    │               └─ Não: NÓ 1869 | debt_to_income_ratio = '39'? | bootstrap: 539 recusados, 17689 aprovados
    │                   ├─ Sim: NÓ 2081 | loan_type = '1'? | bootstrap: 22 recusados, 744 aprovados
    │                   │   ├─ Sim: NÓ 2083 | derived_ethnicity = 'Joint'? | bootstrap: 15 recusados, 544 aprovados
    │                   │   │   ├─ Sim: FOLHA 2091 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
    │                   │   │   └─ Não: NÓ 2084 | applicant_age = '25-34'? | bootstrap: 15 recusados, 527 aprovados
    │                   │   │       ├─ Sim: NÓ 2086 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 6 recusados, 238 aprovados
    │                   │   │       │   ├─ Sim: FOLHA 2090 | previsão: APROVADO (1) | bootstrap: 0 recusados, 21 aprovados
    │                   │   │       │   └─ Não: NÓ 2087 | loan_amount (valor solicitado em USD) <= 895000.0? | bootstrap: 6 recusados, 217 aprovados
    │                   │   │       │       ├─ Sim: FOLHA 2088 | previsão: APROVADO (1) | bootstrap: 4 recusados, 211 aprovados
    │                   │   │       │       └─ Não: FOLHA 2089 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
    │                   │   │       └─ Não: FOLHA 2085 | previsão: APROVADO (1) | bootstrap: 9 recusados, 289 aprovados
    │                   │   └─ Não: FOLHA 2082 | previsão: APROVADO (1) | bootstrap: 7 recusados, 200 aprovados
    │                   └─ Não: NÓ 1870 | total_units = '1'? | bootstrap: 517 recusados, 16945 aprovados
    │                       ├─ Sim: NÓ 1872 | applicant_age = '65-74'? | bootstrap: 496 recusados, 16695 aprovados
    │                       │   ├─ Sim: NÓ 2054 | debt_to_income_ratio = '36'? | bootstrap: 32 recusados, 914 aprovados
    │                       │   │   ├─ Sim: FOLHA 2080 | previsão: APROVADO (1) | bootstrap: 0 recusados, 27 aprovados
    │                       │   │   └─ Não: NÓ 2055 | loan_amount (valor solicitado em USD) <= 440000.0? | bootstrap: 32 recusados, 887 aprovados
    │                       │   │       ├─ Sim: NÓ 2056 | debt_to_income_ratio = '41'? | bootstrap: 32 recusados, 659 aprovados
    │                       │   │       │   ├─ Sim: NÓ 2076 | property_value (valor do imóvel em USD) <= 290000.0? | bootstrap: 2 recusados, 24 aprovados
    │                       │   │       │   │   ├─ Sim: FOLHA 2077 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
    │                       │   │       │   │   └─ Não: FOLHA 2078 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
    │                       │   │       │   └─ Não: NÓ 2057 | derived_ethnicity = 'Joint'? | bootstrap: 30 recusados, 635 aprovados
    │                       │   │       │       ├─ Sim: FOLHA 2075 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
    │                       │   │       │       └─ Não: NÓ 2058 | construction_method = '1'? | bootstrap: 30 recusados, 616 aprovados
    │                       │   │       │           ├─ Sim: NÓ 2060 | loan_type = '2'? | bootstrap: 27 recusados, 596 aprovados
    │                       │   │       │           │   ├─ Sim: FOLHA 2074 | previsão: APROVADO (1) | bootstrap: 11 recusados, 51 aprovados
    │                       │   │       │           │   └─ Não: NÓ 2061 | loan_type = '1'? | bootstrap: 16 recusados, 545 aprovados
    │                       │   │       │           │       ├─ Sim: NÓ 2063 | debt_to_income_ratio = '46'? | bootstrap: 9 recusados, 471 aprovados
    │                       │   │       │           │       │   ├─ Sim: FOLHA 2073 | previsão: APROVADO (1) | bootstrap: 0 recusados, 27 aprovados
    │                       │   │       │           │       │   └─ Não: NÓ 2064 | debt_to_income_ratio = '44'? | bootstrap: 9 recusados, 444 aprovados
    │                       │   │       │           │       │       ├─ Sim: FOLHA 2072 | previsão: APROVADO (1) | bootstrap: 0 recusados, 37 aprovados
    │                       │   │       │           │       │       └─ Não: NÓ 2065 | property_value (valor do imóvel em USD) <= 480000.0? | bootstrap: 9 recusados, 407 aprovados
    │                       │   │       │           │       │           ├─ Sim: NÓ 2066 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 62.0? | bootstrap: 2 recusados, 278 aprovados
    │                       │   │       │           │       │           │   ├─ Sim: FOLHA 2067 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
    │                       │   │       │           │       │           │   └─ Não: FOLHA 2068 | previsão: APROVADO (1) | bootstrap: 1 recusados, 264 aprovados
    │                       │   │       │           │       │           └─ Não: NÓ 2069 | debt_to_income_ratio = '48'? | bootstrap: 7 recusados, 129 aprovados
    │                       │   │       │           │       │               ├─ Sim: FOLHA 2071 | previsão: APROVADO (1) | bootstrap: 2 recusados, 11 aprovados
    │                       │   │       │           │       │               └─ Não: FOLHA 2070 | previsão: APROVADO (1) | bootstrap: 5 recusados, 118 aprovados
    │                       │   │       │           │       └─ Não: FOLHA 2062 | previsão: APROVADO (1) | bootstrap: 7 recusados, 74 aprovados
    │                       │   │       │           └─ Não: FOLHA 2059 | previsão: APROVADO (1) | bootstrap: 3 recusados, 20 aprovados
    │                       │   │       └─ Não: FOLHA 2079 | previsão: APROVADO (1) | bootstrap: 0 recusados, 228 aprovados
    │                       │   └─ Não: NÓ 1873 | debt_to_income_ratio = '49'? | bootstrap: 464 recusados, 15781 aprovados
    │                       │       ├─ Sim: NÓ 2009 | applicant_age = '>74'? | bootstrap: 50 recusados, 1354 aprovados
    │                       │       │   ├─ Sim: NÓ 2049 | property_value (valor do imóvel em USD) <= 445000.0? | bootstrap: 3 recusados, 38 aprovados
    │                       │       │   │   ├─ Sim: NÓ 2050 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 102.5? | bootstrap: 3 recusados, 12 aprovados
    │                       │       │   │   │   ├─ Sim: FOLHA 2051 | previsão: RECUSADO (0) | bootstrap: 3 recusados, 3 aprovados
    │                       │       │   │   │   └─ Não: FOLHA 2052 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
    │                       │       │   │   └─ Não: FOLHA 2053 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
    │                       │       │   └─ Não: NÓ 2010 | loan_type = '1'? | bootstrap: 47 recusados, 1316 aprovados
    │                       │       │       ├─ Sim: NÓ 2038 | applicant_age = '<25'? | bootstrap: 26 recusados, 955 aprovados
    │                       │       │       │   ├─ Sim: NÓ 2046 | loan_amount (valor solicitado em USD) <= 210000.0? | bootstrap: 2 recusados, 59 aprovados
    │                       │       │       │   │   ├─ Sim: FOLHA 2047 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
    │                       │       │       │   │   └─ Não: FOLHA 2048 | previsão: APROVADO (1) | bootstrap: 0 recusados, 50 aprovados
    │                       │       │       │   └─ Não: NÓ 2039 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 24 recusados, 896 aprovados
    │                       │       │       │       ├─ Sim: NÓ 2041 | derived_race = 'Asian'? | bootstrap: 15 recusados, 776 aprovados
    │                       │       │       │       │   ├─ Sim: NÓ 2043 | income (renda anual declarada em milhares de USD) <= 75.5? | bootstrap: 2 recusados, 98 aprovados
    │                       │       │       │       │   │   ├─ Sim: FOLHA 2044 | previsão: APROVADO (1) | bootstrap: 2 recusados, 18 aprovados
    │                       │       │       │       │   │   └─ Não: FOLHA 2045 | previsão: APROVADO (1) | bootstrap: 0 recusados, 80 aprovados
    │                       │       │       │       │   └─ Não: FOLHA 2042 | previsão: APROVADO (1) | bootstrap: 13 recusados, 678 aprovados
    │                       │       │       │       └─ Não: FOLHA 2040 | previsão: APROVADO (1) | bootstrap: 9 recusados, 120 aprovados
    │                       │       │       └─ Não: NÓ 2011 | derived_race = 'Joint'? | bootstrap: 21 recusados, 361 aprovados
    │                       │       │           ├─ Sim: FOLHA 2037 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
    │                       │       │           └─ Não: NÓ 2012 | derived_ethnicity = 'Joint'? | bootstrap: 21 recusados, 347 aprovados
    │                       │       │               ├─ Sim: NÓ 2034 | property_value (valor do imóvel em USD) <= 320000.0? | bootstrap: 2 recusados, 17 aprovados
    │                       │       │               │   ├─ Sim: FOLHA 2035 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
    │                       │       │               │   └─ Não: FOLHA 2036 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
    │                       │       │               └─ Não: NÓ 2013 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 19 recusados, 330 aprovados
    │                       │       │                   ├─ Sim: NÓ 2029 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 133.5? | bootstrap: 5 recusados, 78 aprovados
    │                       │       │                   │   ├─ Sim: NÓ 2030 | property_value (valor do imóvel em USD) <= 360000.0? | bootstrap: 2 recusados, 72 aprovados
    │                       │       │                   │   │   ├─ Sim: FOLHA 2031 | previsão: APROVADO (1) | bootstrap: 0 recusados, 47 aprovados
    │                       │       │                   │   │   └─ Não: FOLHA 2032 | previsão: APROVADO (1) | bootstrap: 2 recusados, 25 aprovados
    │                       │       │                   │   └─ Não: FOLHA 2033 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
    │                       │       │                   └─ Não: NÓ 2014 | derived_race = 'White'? | bootstrap: 14 recusados, 252 aprovados
    │                       │       │                       ├─ Sim: NÓ 2018 | loan_type = '3'? | bootstrap: 11 recusados, 186 aprovados
    │                       │       │                       │   ├─ Sim: FOLHA 2028 | previsão: APROVADO (1) | bootstrap: 0 recusados, 34 aprovados
    │                       │       │                       │   └─ Não: NÓ 2019 | income (renda anual declarada em milhares de USD) <= 139.0? | bootstrap: 11 recusados, 152 aprovados
    │                       │       │                       │       ├─ Sim: NÓ 2020 | loan_amount (valor solicitado em USD) <= 250000.0? | bootstrap: 5 recusados, 135 aprovados
    │                       │       │                       │       │   ├─ Sim: FOLHA 2021 | previsão: APROVADO (1) | bootstrap: 0 recusados, 49 aprovados
    │                       │       │                       │       │   └─ Não: NÓ 2022 | applicant_age = '25-34'? | bootstrap: 5 recusados, 86 aprovados
    │                       │       │                       │       │       ├─ Sim: FOLHA 2024 | previsão: APROVADO (1) | bootstrap: 1 recusados, 52 aprovados
    │                       │       │                       │       │       └─ Não: FOLHA 2023 | previsão: APROVADO (1) | bootstrap: 4 recusados, 34 aprovados
    │                       │       │                       │       └─ Não: NÓ 2025 | applicant_age = '25-34'? | bootstrap: 6 recusados, 17 aprovados
    │                       │       │                       │           ├─ Sim: FOLHA 2027 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │                       │       │                       │           └─ Não: FOLHA 2026 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 4 aprovados
    │                       │       │                       └─ Não: NÓ 2015 | loan_type = '3'? | bootstrap: 3 recusados, 66 aprovados
    │                       │       │                           ├─ Sim: FOLHA 2017 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │                       │       │                           └─ Não: FOLHA 2016 | previsão: APROVADO (1) | bootstrap: 3 recusados, 56 aprovados
    │                       │       └─ Não: NÓ 1874 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 414 recusados, 14427 aprovados
    │                       │           ├─ Sim: NÓ 1920 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 303 recusados, 12443 aprovados
    │                       │           │   ├─ Sim: NÓ 1970 | applicant_age = '35-44'? | bootstrap: 48 recusados, 2681 aprovados
    │                       │           │   │   ├─ Sim: NÓ 1988 | loan_type = '1'? | bootstrap: 14 recusados, 811 aprovados
    │                       │           │   │   │   ├─ Sim: NÓ 1994 | construction_method = '1'? | bootstrap: 12 recusados, 636 aprovados
    │                       │           │   │   │   │   ├─ Sim: NÓ 1996 | derived_race = 'Joint'? | bootstrap: 12 recusados, 628 aprovados
    │                       │           │   │   │   │   │   ├─ Sim: FOLHA 2008 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
    │                       │           │   │   │   │   │   └─ Não: NÓ 1997 | loan_amount (valor solicitado em USD) <= 390000.0? | bootstrap: 12 recusados, 608 aprovados
    │                       │           │   │   │   │   │       ├─ Sim: FOLHA 1998 | previsão: APROVADO (1) | bootstrap: 9 recusados, 306 aprovados
    │                       │           │   │   │   │   │       └─ Não: NÓ 1999 | income (renda anual declarada em milhares de USD) <= 236.5? | bootstrap: 3 recusados, 302 aprovados
    │                       │           │   │   │   │   │           ├─ Sim: FOLHA 2000 | previsão: APROVADO (1) | bootstrap: 0 recusados, 160 aprovados
    │                       │           │   │   │   │   │           └─ Não: NÓ 2001 | derived_race = 'White'? | bootstrap: 3 recusados, 142 aprovados
    │                       │           │   │   │   │   │               ├─ Sim: NÓ 2005 | income (renda anual declarada em milhares de USD) <= 241.0? | bootstrap: 2 recusados, 126 aprovados
    │                       │           │   │   │   │   │               │   ├─ Sim: FOLHA 2006 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
    │                       │           │   │   │   │   │               │   └─ Não: FOLHA 2007 | previsão: APROVADO (1) | bootstrap: 0 recusados, 120 aprovados
    │                       │           │   │   │   │   │               └─ Não: NÓ 2002 | income (renda anual declarada em milhares de USD) <= 304.5? | bootstrap: 1 recusados, 16 aprovados
    │                       │           │   │   │   │   │                   ├─ Sim: FOLHA 2003 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
    │                       │           │   │   │   │   │                   └─ Não: FOLHA 2004 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
    │                       │           │   │   │   │   └─ Não: FOLHA 1995 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │                       │           │   │   │   └─ Não: NÓ 1989 | loan_type = '3'? | bootstrap: 2 recusados, 175 aprovados
    │                       │           │   │   │       ├─ Sim: FOLHA 1993 | previsão: APROVADO (1) | bootstrap: 0 recusados, 62 aprovados
    │                       │           │   │   │       └─ Não: NÓ 1990 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 136.5? | bootstrap: 2 recusados, 113 aprovados
    │                       │           │   │   │           ├─ Sim: FOLHA 1991 | previsão: APROVADO (1) | bootstrap: 1 recusados, 108 aprovados
    │                       │           │   │   │           └─ Não: FOLHA 1992 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
    │                       │           │   │   └─ Não: NÓ 1971 | loan_type = '1'? | bootstrap: 34 recusados, 1870 aprovados
    │                       │           │   │       ├─ Sim: NÓ 1973 | loan_term (prazo em meses) <= 300.0? | bootstrap: 25 recusados, 1433 aprovados
    │                       │           │   │       │   ├─ Sim: FOLHA 1974 | previsão: APROVADO (1) | bootstrap: 0 recusados, 30 aprovados
    │                       │           │   │       │   └─ Não: NÓ 1975 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 277.5? | bootstrap: 25 recusados, 1403 aprovados
    │                       │           │   │       │       ├─ Sim: NÓ 1976 | derived_race = 'Asian'? | bootstrap: 22 recusados, 1396 aprovados
    │                       │           │   │       │       │   ├─ Sim: NÓ 1984 | income (renda anual declarada em milhares de USD) <= 194.0? | bootstrap: 3 recusados, 63 aprovados
    │                       │           │   │       │       │   │   ├─ Sim: FOLHA 1985 | previsão: APROVADO (1) | bootstrap: 0 recusados, 55 aprovados
    │                       │           │   │       │       │   │   └─ Não: FOLHA 1986 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
    │                       │           │   │       │       │   └─ Não: NÓ 1977 | construction_method = '2'? | bootstrap: 19 recusados, 1333 aprovados
    │                       │           │   │       │       │       ├─ Sim: FOLHA 1983 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
    │                       │           │   │       │       │       └─ Não: NÓ 1978 | derived_race = 'Joint'? | bootstrap: 19 recusados, 1316 aprovados
    │                       │           │   │       │       │           ├─ Sim: FOLHA 1982 | previsão: APROVADO (1) | bootstrap: 0 recusados, 59 aprovados
    │                       │           │   │       │       │           └─ Não: NÓ 1979 | property_value (valor do imóvel em USD) <= 570000.0? | bootstrap: 19 recusados, 1257 aprovados
    │                       │           │   │       │       │               ├─ Sim: FOLHA 1980 | previsão: APROVADO (1) | bootstrap: 19 recusados, 977 aprovados
    │                       │           │   │       │       │               └─ Não: FOLHA 1981 | previsão: APROVADO (1) | bootstrap: 0 recusados, 280 aprovados
    │                       │           │   │       │       └─ Não: FOLHA 1987 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
    │                       │           │   │       └─ Não: FOLHA 1972 | previsão: APROVADO (1) | bootstrap: 9 recusados, 437 aprovados
    │                       │           │   └─ Não: NÓ 1921 | debt_to_income_ratio = '47'? | bootstrap: 255 recusados, 9762 aprovados
    │                       │           │       ├─ Sim: NÓ 1955 | applicant_age = '25-34'? | bootstrap: 19 recusados, 645 aprovados
    │                       │           │       │   ├─ Sim: FOLHA 1969 | previsão: APROVADO (1) | bootstrap: 9 recusados, 273 aprovados
    │                       │           │       │   └─ Não: NÓ 1956 | construction_method = '2'? | bootstrap: 10 recusados, 372 aprovados
    │                       │           │       │       ├─ Sim: FOLHA 1968 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │                       │           │       │       └─ Não: NÓ 1957 | applicant_age = '<25'? | bootstrap: 10 recusados, 364 aprovados
    │                       │           │       │           ├─ Sim: FOLHA 1967 | previsão: APROVADO (1) | bootstrap: 2 recusados, 46 aprovados
    │                       │           │       │           └─ Não: NÓ 1958 | loan_amount (valor solicitado em USD) <= 805000.0? | bootstrap: 8 recusados, 318 aprovados
    │                       │           │       │               ├─ Sim: NÓ 1959 | applicant_age = '>74'? | bootstrap: 6 recusados, 310 aprovados
    │                       │           │       │               │   ├─ Sim: FOLHA 1965 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
    │                       │           │       │               │   └─ Não: NÓ 1960 | derived_race = 'White'? | bootstrap: 6 recusados, 291 aprovados
    │                       │           │       │               │       ├─ Sim: FOLHA 1964 | previsão: APROVADO (1) | bootstrap: 0 recusados, 224 aprovados
    │                       │           │       │               │       └─ Não: NÓ 1961 | loan_type = '1'? | bootstrap: 6 recusados, 67 aprovados
    │                       │           │       │               │           ├─ Sim: FOLHA 1963 | previsão: APROVADO (1) | bootstrap: 3 recusados, 42 aprovados
    │                       │           │       │               │           └─ Não: FOLHA 1962 | previsão: APROVADO (1) | bootstrap: 3 recusados, 25 aprovados
    │                       │           │       │               └─ Não: FOLHA 1966 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
    │                       │           │       └─ Não: NÓ 1922 | applicant_age = '55-64'? | bootstrap: 236 recusados, 9117 aprovados
    │                       │           │           ├─ Sim: NÓ 1926 | property_value (valor do imóvel em USD) <= 580000.0? | bootstrap: 36 recusados, 1056 aprovados
    │                       │           │           │   ├─ Sim: NÓ 1927 | derived_race = 'Joint'? | bootstrap: 30 recusados, 767 aprovados
    │                       │           │           │   │   ├─ Sim: FOLHA 1939 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │                       │           │           │   │   └─ Não: NÓ 1928 | construction_method = '1'? | bootstrap: 30 recusados, 759 aprovados
    │                       │           │           │   │       ├─ Sim: NÓ 1930 | debt_to_income_ratio = '<20%'? | bootstrap: 30 recusados, 728 aprovados
    │                       │           │           │   │       │   ├─ Sim: FOLHA 1938 | previsão: APROVADO (1) | bootstrap: 0 recusados, 41 aprovados
    │                       │           │           │   │       │   └─ Não: NÓ 1931 | debt_to_income_ratio = '41'? | bootstrap: 30 recusados, 687 aprovados
    │                       │           │           │   │       │       ├─ Sim: NÓ 1935 | loan_amount (valor solicitado em USD) <= 250000.0? | bootstrap: 2 recusados, 60 aprovados
    │                       │           │           │   │       │       │   ├─ Sim: FOLHA 1936 | previsão: APROVADO (1) | bootstrap: 2 recusados, 21 aprovados
    │                       │           │           │   │       │       │   └─ Não: FOLHA 1937 | previsão: APROVADO (1) | bootstrap: 0 recusados, 39 aprovados
    │                       │           │           │   │       │       └─ Não: NÓ 1932 | debt_to_income_ratio = '48'? | bootstrap: 28 recusados, 627 aprovados
    │                       │           │           │   │       │           ├─ Sim: FOLHA 1934 | previsão: APROVADO (1) | bootstrap: 5 recusados, 57 aprovados
    │                       │           │           │   │       │           └─ Não: FOLHA 1933 | previsão: APROVADO (1) | bootstrap: 23 recusados, 570 aprovados
    │                       │           │           │   │       └─ Não: FOLHA 1929 | previsão: APROVADO (1) | bootstrap: 0 recusados, 31 aprovados
    │                       │           │           │   └─ Não: NÓ 1940 | debt_to_income_ratio = '36'? | bootstrap: 6 recusados, 289 aprovados
    │                       │           │           │       ├─ Sim: NÓ 1952 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 142.5? | bootstrap: 1 recusados, 12 aprovados
    │                       │           │           │       │   ├─ Sim: FOLHA 1953 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
    │                       │           │           │       │   └─ Não: FOLHA 1954 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
    │                       │           │           │       └─ Não: NÓ 1941 | debt_to_income_ratio = '<20%'? | bootstrap: 5 recusados, 277 aprovados
    │                       │           │           │           ├─ Sim: FOLHA 1951 | previsão: APROVADO (1) | bootstrap: 3 recusados, 38 aprovados
    │                       │           │           │           └─ Não: NÓ 1942 | property_value (valor do imóvel em USD) <= 1265000.0? | bootstrap: 2 recusados, 239 aprovados
    │                       │           │           │               ├─ Sim: NÓ 1943 | loan_type = '3'? | bootstrap: 1 recusados, 220 aprovados
    │                       │           │           │               │   ├─ Sim: FOLHA 1947 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
    │                       │           │           │               │   └─ Não: NÓ 1944 | debt_to_income_ratio = '43'? | bootstrap: 1 recusados, 204 aprovados
    │                       │           │           │               │       ├─ Sim: FOLHA 1946 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
    │                       │           │           │               │       └─ Não: FOLHA 1945 | previsão: APROVADO (1) | bootstrap: 1 recusados, 187 aprovados
    │                       │           │           │               └─ Não: NÓ 1948 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 1 recusados, 19 aprovados
    │                       │           │           │                   ├─ Sim: FOLHA 1950 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │                       │           │           │                   └─ Não: FOLHA 1949 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
    │                       │           │           └─ Não: NÓ 1923 | derived_race = '2 or more minority races'? | bootstrap: 200 recusados, 8061 aprovados
    │                       │           │               ├─ Sim: FOLHA 1925 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
    │                       │           │               └─ Não: FOLHA 1924 | previsão: APROVADO (1) | bootstrap: 200 recusados, 8049 aprovados
    │                       │           └─ Não: NÓ 1875 | loan_type = '4'? | bootstrap: 111 recusados, 1984 aprovados
    │                       │               ├─ Sim: FOLHA 1919 | previsão: APROVADO (1) | bootstrap: 2 recusados, 18 aprovados
    │                       │               └─ Não: NÓ 1876 | debt_to_income_ratio = '47'? | bootstrap: 109 recusados, 1966 aprovados
    │                       │                   ├─ Sim: NÓ 1914 | loan_type = '1'? | bootstrap: 8 recusados, 152 aprovados
    │                       │                   │   ├─ Sim: FOLHA 1918 | previsão: APROVADO (1) | bootstrap: 0 recusados, 90 aprovados
    │                       │                   │   └─ Não: NÓ 1915 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 119.0? | bootstrap: 8 recusados, 62 aprovados
    │                       │                   │       ├─ Sim: FOLHA 1916 | previsão: APROVADO (1) | bootstrap: 8 recusados, 44 aprovados
    │                       │                   │       └─ Não: FOLHA 1917 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
    │                       │                   └─ Não: NÓ 1877 | debt_to_income_ratio = '41'? | bootstrap: 101 recusados, 1814 aprovados
    │                       │                       ├─ Sim: NÓ 1903 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 7 recusados, 135 aprovados
    │                       │                       │   ├─ Sim: NÓ 1905 | loan_amount (valor solicitado em USD) <= 350000.0? | bootstrap: 7 recusados, 99 aprovados
    │                       │                       │   │   ├─ Sim: NÓ 1906 | loan_type = '1'? | bootstrap: 2 recusados, 56 aprovados
    │                       │                       │   │   │   ├─ Sim: FOLHA 1912 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
    │                       │                       │   │   │   └─ Não: NÓ 1907 | applicant_age = '35-44'? | bootstrap: 2 recusados, 31 aprovados
    │                       │                       │   │   │       ├─ Sim: FOLHA 1911 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
    │                       │                       │   │   │       └─ Não: NÓ 1908 | income (renda anual declarada em milhares de USD) <= 91.0? | bootstrap: 2 recusados, 21 aprovados
    │                       │                       │   │   │           ├─ Sim: FOLHA 1909 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
    │                       │                       │   │   │           └─ Não: FOLHA 1910 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
    │                       │                       │   │   └─ Não: FOLHA 1913 | previsão: APROVADO (1) | bootstrap: 5 recusados, 43 aprovados
    │                       │                       │   └─ Não: FOLHA 1904 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
    │                       │                       └─ Não: NÓ 1878 | property_value (valor do imóvel em USD) <= 1230000.0? | bootstrap: 94 recusados, 1679 aprovados
    │                       │                           ├─ Sim: NÓ 1879 | loan_type = '3'? | bootstrap: 88 recusados, 1664 aprovados
    │                       │                           │   ├─ Sim: NÓ 1897 | income (renda anual declarada em milhares de USD) <= 126.0? | bootstrap: 2 recusados, 146 aprovados
    │                       │                           │   │   ├─ Sim: FOLHA 1898 | previsão: APROVADO (1) | bootstrap: 0 recusados, 80 aprovados
    │                       │                           │   │   └─ Não: FOLHA 1899 | previsão: APROVADO (1) | bootstrap: 2 recusados, 66 aprovados
    │                       │                           │   └─ Não: NÓ 1880 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 86 recusados, 1518 aprovados
    │                       │                           │       ├─ Sim: NÓ 1894 | income (renda anual declarada em milhares de USD) <= 108.0? | bootstrap: 6 recusados, 289 aprovados
    │                       │                           │       │   ├─ Sim: FOLHA 1895 | previsão: APROVADO (1) | bootstrap: 6 recusados, 93 aprovados
    │                       │                           │       │   └─ Não: FOLHA 1896 | previsão: APROVADO (1) | bootstrap: 0 recusados, 196 aprovados
    │                       │                           │       └─ Não: NÓ 1881 | debt_to_income_ratio = '40'? | bootstrap: 80 recusados, 1229 aprovados
    │                       │                           │           ├─ Sim: NÓ 1889 | derived_ethnicity = 'Joint'? | bootstrap: 1 recusados, 119 aprovados
    │                       │                           │           │   ├─ Sim: FOLHA 1893 | previsão: APROVADO (1) | bootstrap: 0 recusados, 39 aprovados
    │                       │                           │           │   └─ Não: NÓ 1890 | applicant_age = '<25'? | bootstrap: 1 recusados, 80 aprovados
    │                       │                           │           │       ├─ Sim: FOLHA 1892 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │                       │                           │           │       └─ Não: FOLHA 1891 | previsão: APROVADO (1) | bootstrap: 1 recusados, 72 aprovados
    │                       │                           │           └─ Não: NÓ 1882 | applicant_age = '35-44'? | bootstrap: 79 recusados, 1110 aprovados
    │                       │                           │               ├─ Sim: NÓ 1886 | derived_race = 'White'? | bootstrap: 20 recusados, 353 aprovados
    │                       │                           │               │   ├─ Sim: FOLHA 1888 | previsão: APROVADO (1) | bootstrap: 18 recusados, 316 aprovados
    │                       │                           │               │   └─ Não: FOLHA 1887 | previsão: APROVADO (1) | bootstrap: 2 recusados, 37 aprovados
    │                       │                           │               └─ Não: NÓ 1883 | applicant_age = '>74'? | bootstrap: 59 recusados, 757 aprovados
    │                       │                           │                   ├─ Sim: FOLHA 1885 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
    │                       │                           │                   └─ Não: FOLHA 1884 | previsão: APROVADO (1) | bootstrap: 59 recusados, 749 aprovados
    │                       │                           └─ Não: NÓ 1900 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 149.5? | bootstrap: 6 recusados, 15 aprovados
    │                       │                               ├─ Sim: FOLHA 1901 | previsão: APROVADO (1) | bootstrap: 4 recusados, 6 aprovados
    │                       │                               └─ Não: FOLHA 1902 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
    │                       └─ Não: FOLHA 1871 | previsão: APROVADO (1) | bootstrap: 21 recusados, 250 aprovados
    └─ Não: NÓ 2 | debt_to_income_ratio = '30%-<36%'? | bootstrap: 166481 recusados, 1276070 aprovados
        ├─ Sim: NÓ 1378 | lei = '254900ZFWS2106HWPH46'? | bootstrap: 13903 recusados, 195887 aprovados
        │   ├─ Sim: FOLHA 1628 | previsão: APROVADO (1) | bootstrap: 7 recusados, 757 aprovados
        │   └─ Não: NÓ 1379 | lei = '549300R77W0P105C3S50'? | bootstrap: 13896 recusados, 195130 aprovados
        │       ├─ Sim: NÓ 1619 | applicant_age = '35-44'? | bootstrap: 31 recusados, 37 aprovados
        │       │   ├─ Sim: NÓ 1621 | derived_race = 'White'? | bootstrap: 12 recusados, 20 aprovados
        │       │   │   ├─ Sim: NÓ 1623 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 152.0? | bootstrap: 11 recusados, 12 aprovados
        │       │   │   │   ├─ Sim: NÓ 1624 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 87.0? | bootstrap: 9 recusados, 6 aprovados
        │       │   │   │   │   ├─ Sim: FOLHA 1625 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 4 aprovados
        │       │   │   │   │   └─ Não: FOLHA 1626 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 2 aprovados
        │       │   │   │   └─ Não: FOLHA 1627 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
        │       │   │   └─ Não: FOLHA 1622 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
        │       │   └─ Não: FOLHA 1620 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 17 aprovados
        │       └─ Não: NÓ 1380 | lei = '254900DTLHVWQ7NP7R34'? | bootstrap: 13865 recusados, 195093 aprovados
        │           ├─ Sim: NÓ 1616 | loan_amount (valor solicitado em USD) <= 400000.0? | bootstrap: 12 recusados, 759 aprovados
        │           │   ├─ Sim: FOLHA 1617 | previsão: APROVADO (1) | bootstrap: 1 recusados, 443 aprovados
        │           │   └─ Não: FOLHA 1618 | previsão: APROVADO (1) | bootstrap: 11 recusados, 316 aprovados
        │           └─ Não: NÓ 1381 | derived_race = 'Asian'? | bootstrap: 13853 recusados, 194334 aprovados
        │               ├─ Sim: NÓ 1563 | income (renda anual declarada em milhares de USD) <= 74.5? | bootstrap: 836 recusados, 16644 aprovados
        │               │   ├─ Sim: NÓ 1564 | lei = '549300RJW4T6MPVUPK44'? | bootstrap: 113 recusados, 1112 aprovados
        │               │   │   ├─ Sim: FOLHA 1568 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
        │               │   │   └─ Não: NÓ 1565 | income (renda anual declarada em milhares de USD) <= 24.5? | bootstrap: 113 recusados, 1104 aprovados
        │               │   │       ├─ Sim: FOLHA 1566 | previsão: APROVADO (1) | bootstrap: 0 recusados, 85 aprovados
        │               │   │       └─ Não: FOLHA 1567 | previsão: APROVADO (1) | bootstrap: 113 recusados, 1019 aprovados
        │               │   └─ Não: NÓ 1569 | lei = '549300DD4R4SYK5RAQ92'? | bootstrap: 723 recusados, 15532 aprovados
        │               │       ├─ Sim: NÓ 1613 | loan_type = '3'? | bootstrap: 2 recusados, 229 aprovados
        │               │       │   ├─ Sim: FOLHA 1615 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
        │               │       │   └─ Não: FOLHA 1614 | previsão: APROVADO (1) | bootstrap: 2 recusados, 218 aprovados
        │               │       └─ Não: NÓ 1570 | lei = '549300X8AUU3UC1ULZ79'? | bootstrap: 721 recusados, 15303 aprovados
        │               │           ├─ Sim: NÓ 1610 | property_value (valor do imóvel em USD) <= 815000.0? | bootstrap: 2 recusados, 11 aprovados
        │               │           │   ├─ Sim: FOLHA 1611 | previsão: APROVADO (1) | bootstrap: 0 recusados, 5 aprovados
        │               │           │   └─ Não: FOLHA 1612 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
        │               │           └─ Não: NÓ 1571 | lei = '549300J7I82PNDVU8H22'? | bootstrap: 719 recusados, 15292 aprovados
        │               │               ├─ Sim: FOLHA 1609 | previsão: APROVADO (1) | bootstrap: 4 recusados, 11 aprovados
        │               │               └─ Não: NÓ 1572 | lei = '254900VELJPWWT51WN03'? | bootstrap: 715 recusados, 15281 aprovados
        │               │                   ├─ Sim: FOLHA 1608 | previsão: APROVADO (1) | bootstrap: 0 recusados, 61 aprovados
        │               │                   └─ Não: NÓ 1573 | loan_type = '1'? | bootstrap: 715 recusados, 15220 aprovados
        │               │                       ├─ Sim: NÓ 1591 | lei = '549300OPCWU6E72WUT29'? | bootstrap: 643 recusados, 14089 aprovados
        │               │                       │   ├─ Sim: FOLHA 1607 | previsão: APROVADO (1) | bootstrap: 0 recusados, 65 aprovados
        │               │                       │   └─ Não: NÓ 1592 | lei = '549300Y5Y88FQLI7VE83'? | bootstrap: 643 recusados, 14024 aprovados
        │               │                       │       ├─ Sim: FOLHA 1606 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
        │               │                       │       └─ Não: NÓ 1593 | lei = '5493004N9PMBSLEZOF16'? | bootstrap: 640 recusados, 14019 aprovados
        │               │                       │           ├─ Sim: FOLHA 1605 | previsão: APROVADO (1) | bootstrap: 0 recusados, 50 aprovados
        │               │                       │           └─ Não: NÓ 1594 | lei = 'SS1TRMSN6BRNMOREEV51'? | bootstrap: 640 recusados, 13969 aprovados
        │               │                       │               ├─ Sim: NÓ 1602 | loan_term (prazo em meses) <= 300.0? | bootstrap: 16 recusados, 48 aprovados
        │               │                       │               │   ├─ Sim: FOLHA 1603 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
        │               │                       │               │   └─ Não: FOLHA 1604 | previsão: APROVADO (1) | bootstrap: 13 recusados, 36 aprovados
        │               │                       │               └─ Não: NÓ 1595 | lei = '549300K5OUR2FD1NV703'? | bootstrap: 624 recusados, 13921 aprovados
        │               │                       │                   ├─ Sim: FOLHA 1601 | previsão: APROVADO (1) | bootstrap: 0 recusados, 32 aprovados
        │               │                       │                   └─ Não: NÓ 1596 | lei = '549300BM4NS8HDJT3X28'? | bootstrap: 624 recusados, 13889 aprovados
        │               │                       │                       ├─ Sim: FOLHA 1600 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
        │               │                       │                       └─ Não: NÓ 1597 | lei = '549300PC4MFWQBNVKG88'? | bootstrap: 624 recusados, 13864 aprovados
        │               │                       │                           ├─ Sim: FOLHA 1599 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
        │               │                       │                           └─ Não: FOLHA 1598 | previsão: APROVADO (1) | bootstrap: 624 recusados, 13838 aprovados
        │               │                       └─ Não: NÓ 1574 | lei = '5493004N9PMBSLEZOF16'? | bootstrap: 72 recusados, 1131 aprovados
        │               │                           ├─ Sim: FOLHA 1590 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
        │               │                           └─ Não: NÓ 1575 | lei = '549300GKFNPRWNS0GF29'? | bootstrap: 70 recusados, 1122 aprovados
        │               │                               ├─ Sim: FOLHA 1589 | previsão: APROVADO (1) | bootstrap: 3 recusados, 17 aprovados
        │               │                               └─ Não: NÓ 1576 | total_units = '2'? | bootstrap: 67 recusados, 1105 aprovados
        │               │                                   ├─ Sim: NÓ 1586 | applicant_age = '35-44'? | bootstrap: 2 recusados, 19 aprovados
        │               │                                   │   ├─ Sim: FOLHA 1588 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
        │               │                                   │   └─ Não: FOLHA 1587 | previsão: APROVADO (1) | bootstrap: 2 recusados, 11 aprovados
        │               │                                   └─ Não: NÓ 1577 | lei = '549300LEOR2LO8J9RT29'? | bootstrap: 65 recusados, 1086 aprovados
        │               │                                       ├─ Sim: FOLHA 1585 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
        │               │                                       └─ Não: NÓ 1578 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 65 recusados, 1079 aprovados
        │               │                                           ├─ Sim: FOLHA 1584 | previsão: APROVADO (1) | bootstrap: 7 recusados, 58 aprovados
        │               │                                           └─ Não: NÓ 1579 | loan_amount (valor solicitado em USD) <= 270000.0? | bootstrap: 58 recusados, 1021 aprovados
        │               │                                               ├─ Sim: FOLHA 1580 | previsão: APROVADO (1) | bootstrap: 19 recusados, 125 aprovados
        │               │                                               └─ Não: NÓ 1581 | lei = '5493001SXWZ4OFP8Z903'? | bootstrap: 39 recusados, 896 aprovados
        │               │                                                   ├─ Sim: FOLHA 1583 | previsão: APROVADO (1) | bootstrap: 2 recusados, 97 aprovados
        │               │                                                   └─ Não: FOLHA 1582 | previsão: APROVADO (1) | bootstrap: 37 recusados, 799 aprovados
        │               └─ Não: NÓ 1382 | lei = 'C5654JQHZUHN0772B561'? | bootstrap: 13017 recusados, 177690 aprovados
        │                   ├─ Sim: NÓ 1494 | derived_ethnicity = 'Joint'? | bootstrap: 222 recusados, 845 aprovados
        │                   │   ├─ Sim: NÓ 1558 | derived_race = 'White'? | bootstrap: 9 recusados, 61 aprovados
        │                   │   │   ├─ Sim: NÓ 1560 | loan_type = '1'? | bootstrap: 6 recusados, 45 aprovados
        │                   │   │   │   ├─ Sim: FOLHA 1562 | previsão: APROVADO (1) | bootstrap: 2 recusados, 17 aprovados
        │                   │   │   │   └─ Não: FOLHA 1561 | previsão: APROVADO (1) | bootstrap: 4 recusados, 28 aprovados
        │                   │   │   └─ Não: FOLHA 1559 | previsão: APROVADO (1) | bootstrap: 3 recusados, 16 aprovados
        │                   │   └─ Não: NÓ 1495 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 213 recusados, 784 aprovados
        │                   │       ├─ Sim: FOLHA 1557 | previsão: APROVADO (1) | bootstrap: 4 recusados, 9 aprovados
        │                   │       └─ Não: NÓ 1496 | applicant_age = '65-74'? | bootstrap: 209 recusados, 775 aprovados
        │                   │           ├─ Sim: NÓ 1550 | loan_type = '3'? | bootstrap: 16 recusados, 53 aprovados
        │                   │           │   ├─ Sim: NÓ 1554 | derived_race = 'Black or African American'? | bootstrap: 14 recusados, 32 aprovados
        │                   │           │   │   ├─ Sim: FOLHA 1556 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
        │                   │           │   │   └─ Não: FOLHA 1555 | previsão: APROVADO (1) | bootstrap: 10 recusados, 27 aprovados
        │                   │           │   └─ Não: NÓ 1551 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 92.5? | bootstrap: 2 recusados, 21 aprovados
        │                   │           │       ├─ Sim: FOLHA 1552 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
        │                   │           │       └─ Não: FOLHA 1553 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
        │                   │           └─ Não: NÓ 1497 | derived_race = 'White'? | bootstrap: 193 recusados, 722 aprovados
        │                   │               ├─ Sim: NÓ 1499 | applicant_age = '55-64'? | bootstrap: 162 recusados, 638 aprovados
        │                   │               │   ├─ Sim: NÓ 1529 | property_value (valor do imóvel em USD) <= 150000.0? | bootstrap: 30 recusados, 113 aprovados
        │                   │               │   │   ├─ Sim: FOLHA 1530 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 4 aprovados
        │                   │               │   │   └─ Não: NÓ 1531 | income (renda anual declarada em milhares de USD) <= 51.5? | bootstrap: 26 recusados, 109 aprovados
        │                   │               │   │       ├─ Sim: FOLHA 1532 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 3 aprovados
        │                   │               │   │       └─ Não: NÓ 1533 | property_value (valor do imóvel em USD) <= 300000.0? | bootstrap: 20 recusados, 106 aprovados
        │                   │               │   │           ├─ Sim: NÓ 1534 | loan_type = '1'? | bootstrap: 1 recusados, 27 aprovados
        │                   │               │   │           │   ├─ Sim: FOLHA 1538 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
        │                   │               │   │           │   └─ Não: NÓ 1535 | loan_amount (valor solicitado em USD) <= 185000.0? | bootstrap: 1 recusados, 20 aprovados
        │                   │               │   │           │       ├─ Sim: FOLHA 1536 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
        │                   │               │   │           │       └─ Não: FOLHA 1537 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
        │                   │               │   │           └─ Não: NÓ 1539 | loan_type = '3'? | bootstrap: 19 recusados, 79 aprovados
        │                   │               │   │               ├─ Sim: NÓ 1545 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 96.0? | bootstrap: 10 recusados, 52 aprovados
        │                   │               │   │               │   ├─ Sim: NÓ 1546 | loan_amount (valor solicitado em USD) <= 420000.0? | bootstrap: 7 recusados, 8 aprovados
        │                   │               │   │               │   │   ├─ Sim: FOLHA 1547 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
        │                   │               │   │               │   │   └─ Não: FOLHA 1548 | previsão: APROVADO (1) | bootstrap: 2 recusados, 3 aprovados
        │                   │               │   │               │   └─ Não: FOLHA 1549 | previsão: APROVADO (1) | bootstrap: 3 recusados, 44 aprovados
        │                   │               │   │               └─ Não: NÓ 1540 | property_value (valor do imóvel em USD) <= 390000.0? | bootstrap: 9 recusados, 27 aprovados
        │                   │               │   │                   ├─ Sim: FOLHA 1541 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
        │                   │               │   │                   └─ Não: NÓ 1542 | loan_amount (valor solicitado em USD) <= 390000.0? | bootstrap: 5 recusados, 22 aprovados
        │                   │               │   │                       ├─ Sim: FOLHA 1543 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
        │                   │               │   │                       └─ Não: FOLHA 1544 | previsão: APROVADO (1) | bootstrap: 5 recusados, 7 aprovados
        │                   │               │   └─ Não: NÓ 1500 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 81.5? | bootstrap: 132 recusados, 525 aprovados
        │                   │               │       ├─ Sim: FOLHA 1501 | previsão: APROVADO (1) | bootstrap: 41 recusados, 100 aprovados
        │                   │               │       └─ Não: NÓ 1502 | applicant_age = '>74'? | bootstrap: 91 recusados, 425 aprovados
        │                   │               │           ├─ Sim: NÓ 1526 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 101.0? | bootstrap: 3 recusados, 18 aprovados
        │                   │               │           │   ├─ Sim: FOLHA 1527 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
        │                   │               │           │   └─ Não: FOLHA 1528 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
        │                   │               │           └─ Não: NÓ 1503 | applicant_age = '<25'? | bootstrap: 88 recusados, 407 aprovados
        │                   │               │               ├─ Sim: NÓ 1523 | loan_type = '1'? | bootstrap: 1 recusados, 25 aprovados
        │                   │               │               │   ├─ Sim: FOLHA 1525 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
        │                   │               │               │   └─ Não: FOLHA 1524 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
        │                   │               │               └─ Não: NÓ 1504 | loan_type = '3'? | bootstrap: 87 recusados, 382 aprovados
        │                   │               │                   ├─ Sim: NÓ 1510 | applicant_age = '25-34'? | bootstrap: 42 recusados, 258 aprovados
        │                   │               │                   │   ├─ Sim: NÓ 1518 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 148.5? | bootstrap: 13 recusados, 92 aprovados
        │                   │               │                   │   │   ├─ Sim: NÓ 1519 | property_value (valor do imóvel em USD) <= 260000.0? | bootstrap: 8 recusados, 84 aprovados
        │                   │               │                   │   │   │   ├─ Sim: FOLHA 1520 | previsão: APROVADO (1) | bootstrap: 5 recusados, 20 aprovados
        │                   │               │                   │   │   │   └─ Não: FOLHA 1521 | previsão: APROVADO (1) | bootstrap: 3 recusados, 64 aprovados
        │                   │               │                   │   │   └─ Não: FOLHA 1522 | previsão: APROVADO (1) | bootstrap: 5 recusados, 8 aprovados
        │                   │               │                   │   └─ Não: NÓ 1511 | loan_amount (valor solicitado em USD) <= 410000.0? | bootstrap: 29 recusados, 166 aprovados
        │                   │               │                   │       ├─ Sim: NÓ 1512 | property_value (valor do imóvel em USD) <= 390000.0? | bootstrap: 8 recusados, 109 aprovados
        │                   │               │                   │       │   ├─ Sim: FOLHA 1513 | previsão: APROVADO (1) | bootstrap: 3 recusados, 85 aprovados
        │                   │               │                   │       │   └─ Não: FOLHA 1514 | previsão: APROVADO (1) | bootstrap: 5 recusados, 24 aprovados
        │                   │               │                   │       └─ Não: NÓ 1515 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 118.0? | bootstrap: 21 recusados, 57 aprovados
        │                   │               │                   │           ├─ Sim: FOLHA 1516 | previsão: APROVADO (1) | bootstrap: 6 recusados, 32 aprovados
        │                   │               │                   │           └─ Não: FOLHA 1517 | previsão: APROVADO (1) | bootstrap: 15 recusados, 25 aprovados
        │                   │               │                   └─ Não: NÓ 1505 | applicant_age = '35-44'? | bootstrap: 45 recusados, 124 aprovados
        │                   │               │                       ├─ Sim: FOLHA 1509 | previsão: APROVADO (1) | bootstrap: 15 recusados, 63 aprovados
        │                   │               │                       └─ Não: NÓ 1506 | loan_amount (valor solicitado em USD) <= 485000.0? | bootstrap: 30 recusados, 61 aprovados
        │                   │               │                           ├─ Sim: FOLHA 1507 | previsão: APROVADO (1) | bootstrap: 26 recusados, 59 aprovados
        │                   │               │                           └─ Não: FOLHA 1508 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 2 aprovados
        │                   │               └─ Não: FOLHA 1498 | previsão: APROVADO (1) | bootstrap: 31 recusados, 84 aprovados
        │                   └─ Não: NÓ 1383 | lei = '549300KJZ82173UB3I21'? | bootstrap: 12795 recusados, 176845 aprovados
        │                       ├─ Sim: NÓ 1489 | loan_amount (valor solicitado em USD) <= 130000.0? | bootstrap: 13 recusados, 60 aprovados
        │                       │   ├─ Sim: FOLHA 1490 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 9 aprovados
        │                       │   └─ Não: NÓ 1491 | property_value (valor do imóvel em USD) <= 200000.0? | bootstrap: 3 recusados, 51 aprovados
        │                       │       ├─ Sim: FOLHA 1492 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
        │                       │       └─ Não: FOLHA 1493 | previsão: APROVADO (1) | bootstrap: 0 recusados, 42 aprovados
        │                       └─ Não: NÓ 1384 | lei = '549300RJW4T6MPVUPK44'? | bootstrap: 12782 recusados, 176785 aprovados
        │                           ├─ Sim: NÓ 1486 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 61.5? | bootstrap: 2 recusados, 181 aprovados
        │                           │   ├─ Sim: FOLHA 1487 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
        │                           │   └─ Não: FOLHA 1488 | previsão: APROVADO (1) | bootstrap: 1 recusados, 172 aprovados
        │                           └─ Não: NÓ 1385 | construction_method = '2'? | bootstrap: 12780 recusados, 176604 aprovados
        │                               ├─ Sim: NÓ 1483 | loan_term (prazo em meses) <= 301.5? | bootstrap: 5642 recusados, 9714 aprovados
        │                               │   ├─ Sim: FOLHA 1484 | previsão: APROVADO (1) | bootstrap: 4923 recusados, 5786 aprovados
        │                               │   └─ Não: FOLHA 1485 | previsão: APROVADO (1) | bootstrap: 719 recusados, 3928 aprovados
        │                               └─ Não: NÓ 1386 | lei = '549300FX7K8PTEQUU487'? | bootstrap: 7138 recusados, 166890 aprovados
        │                                   ├─ Sim: NÓ 1464 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 86 recusados, 171 aprovados
        │                                   │   ├─ Sim: NÓ 1466 | loan_term (prazo em meses) <= 341.5? | bootstrap: 61 recusados, 143 aprovados
        │                                   │   │   ├─ Sim: FOLHA 1467 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
        │                                   │   │   └─ Não: NÓ 1468 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 87.5? | bootstrap: 60 recusados, 135 aprovados
        │                                   │   │       ├─ Sim: FOLHA 1469 | previsão: APROVADO (1) | bootstrap: 5 recusados, 34 aprovados
        │                                   │   │       └─ Não: NÓ 1470 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 152.0? | bootstrap: 55 recusados, 101 aprovados
        │                                   │   │           ├─ Sim: NÓ 1471 | applicant_age = '55-64'? | bootstrap: 40 recusados, 85 aprovados
        │                                   │   │           │   ├─ Sim: FOLHA 1481 | previsão: APROVADO (1) | bootstrap: 5 recusados, 15 aprovados
        │                                   │   │           │   └─ Não: NÓ 1472 | applicant_age = '25-34'? | bootstrap: 35 recusados, 70 aprovados
        │                                   │   │           │       ├─ Sim: NÓ 1478 | loan_amount (valor solicitado em USD) <= 370000.0? | bootstrap: 17 recusados, 18 aprovados
        │                                   │   │           │       │   ├─ Sim: FOLHA 1479 | previsão: APROVADO (1) | bootstrap: 11 recusados, 18 aprovados
        │                                   │   │           │       │   └─ Não: FOLHA 1480 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 0 aprovados
        │                                   │   │           │       └─ Não: NÓ 1473 | loan_amount (valor solicitado em USD) <= 425000.0? | bootstrap: 18 recusados, 52 aprovados
        │                                   │   │           │           ├─ Sim: FOLHA 1474 | previsão: APROVADO (1) | bootstrap: 13 recusados, 32 aprovados
        │                                   │   │           │           └─ Não: NÓ 1475 | derived_race = 'White'? | bootstrap: 5 recusados, 20 aprovados
        │                                   │   │           │               ├─ Sim: FOLHA 1477 | previsão: APROVADO (1) | bootstrap: 4 recusados, 12 aprovados
        │                                   │   │           │               └─ Não: FOLHA 1476 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
        │                                   │   │           └─ Não: FOLHA 1482 | previsão: APROVADO (1) | bootstrap: 15 recusados, 16 aprovados
        │                                   │   └─ Não: FOLHA 1465 | previsão: APROVADO (1) | bootstrap: 25 recusados, 28 aprovados
        │                                   └─ Não: NÓ 1387 | lei = 'EQTWLK1G7ODGC2MGLV11'? | bootstrap: 7052 recusados, 166719 aprovados
        │                                       ├─ Sim: NÓ 1443 | loan_type = '4'? | bootstrap: 54 recusados, 660 aprovados
        │                                       │   ├─ Sim: NÓ 1461 | derived_race = 'White'? | bootstrap: 5 recusados, 26 aprovados
        │                                       │   │   ├─ Sim: FOLHA 1463 | previsão: APROVADO (1) | bootstrap: 4 recusados, 19 aprovados
        │                                       │   │   └─ Não: FOLHA 1462 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
        │                                       │   └─ Não: NÓ 1444 | derived_race = 'Joint'? | bootstrap: 49 recusados, 634 aprovados
        │                                       │       ├─ Sim: FOLHA 1460 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
        │                                       │       └─ Não: NÓ 1445 | applicant_age = '>74'? | bootstrap: 49 recusados, 621 aprovados
        │                                       │           ├─ Sim: FOLHA 1459 | previsão: APROVADO (1) | bootstrap: 2 recusados, 39 aprovados
        │                                       │           └─ Não: NÓ 1446 | income (renda anual declarada em milhares de USD) <= 49.0? | bootstrap: 47 recusados, 582 aprovados
        │                                       │               ├─ Sim: NÓ 1447 | loan_amount (valor solicitado em USD) <= 100000.0? | bootstrap: 13 recusados, 44 aprovados
        │                                       │               │   ├─ Sim: FOLHA 1448 | previsão: APROVADO (1) | bootstrap: 10 recusados, 19 aprovados
        │                                       │               │   └─ Não: FOLHA 1449 | previsão: APROVADO (1) | bootstrap: 3 recusados, 25 aprovados
        │                                       │               └─ Não: NÓ 1450 | derived_race = 'Black or African American'? | bootstrap: 34 recusados, 538 aprovados
        │                                       │                   ├─ Sim: FOLHA 1458 | previsão: APROVADO (1) | bootstrap: 9 recusados, 50 aprovados
        │                                       │                   └─ Não: NÓ 1451 | loan_term (prazo em meses) <= 264.0? | bootstrap: 25 recusados, 488 aprovados
        │                                       │                       ├─ Sim: NÓ 1452 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 112.5? | bootstrap: 5 recusados, 24 aprovados
        │                                       │                       │   ├─ Sim: FOLHA 1453 | previsão: APROVADO (1) | bootstrap: 5 recusados, 7 aprovados
        │                                       │                       │   └─ Não: FOLHA 1454 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
        │                                       │                       └─ Não: NÓ 1455 | loan_amount (valor solicitado em USD) <= 1115000.0? | bootstrap: 20 recusados, 464 aprovados
        │                                       │                           ├─ Sim: FOLHA 1456 | previsão: APROVADO (1) | bootstrap: 14 recusados, 438 aprovados
        │                                       │                           └─ Não: FOLHA 1457 | previsão: APROVADO (1) | bootstrap: 6 recusados, 26 aprovados
        │                                       └─ Não: NÓ 1388 | lei = '549300R9S3MVDV4MGF56'? | bootstrap: 6998 recusados, 166059 aprovados
        │                                           ├─ Sim: FOLHA 1442 | previsão: APROVADO (1) | bootstrap: 25 recusados, 157 aprovados
        │                                           └─ Não: NÓ 1389 | lei = 'SS1TRMSN6BRNMOREEV51'? | bootstrap: 6973 recusados, 165902 aprovados
        │                                               ├─ Sim: NÓ 1421 | derived_race = 'Joint'? | bootstrap: 201 recusados, 1132 aprovados
        │                                               │   ├─ Sim: NÓ 1439 | loan_type = '2'? | bootstrap: 13 recusados, 58 aprovados
        │                                               │   │   ├─ Sim: FOLHA 1441 | previsão: APROVADO (1) | bootstrap: 4 recusados, 21 aprovados
        │                                               │   │   └─ Não: FOLHA 1440 | previsão: APROVADO (1) | bootstrap: 9 recusados, 37 aprovados
        │                                               │   └─ Não: NÓ 1422 | total_units = '1'? | bootstrap: 188 recusados, 1074 aprovados
        │                                               │       ├─ Sim: NÓ 1424 | applicant_age = '35-44'? | bootstrap: 182 recusados, 1071 aprovados
        │                                               │       │   ├─ Sim: NÓ 1432 | derived_race = 'Black or African American'? | bootstrap: 59 recusados, 247 aprovados
        │                                               │       │   │   ├─ Sim: NÓ 1436 | loan_amount (valor solicitado em USD) <= 240000.0? | bootstrap: 1 recusados, 20 aprovados
        │                                               │       │   │   │   ├─ Sim: FOLHA 1437 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
        │                                               │       │   │   │   └─ Não: FOLHA 1438 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
        │                                               │       │   │   └─ Não: NÓ 1433 | property_value (valor do imóvel em USD) <= 110000.0? | bootstrap: 58 recusados, 227 aprovados
        │                                               │       │   │       ├─ Sim: FOLHA 1434 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 2 aprovados
        │                                               │       │   │       └─ Não: FOLHA 1435 | previsão: APROVADO (1) | bootstrap: 53 recusados, 225 aprovados
        │                                               │       │   └─ Não: NÓ 1425 | property_value (valor do imóvel em USD) <= 750000.0? | bootstrap: 123 recusados, 824 aprovados
        │                                               │       │       ├─ Sim: NÓ 1426 | loan_term (prazo em meses) <= 210.0? | bootstrap: 98 recusados, 762 aprovados
        │                                               │       │       │   ├─ Sim: FOLHA 1427 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
        │                                               │       │       │   └─ Não: FOLHA 1428 | previsão: APROVADO (1) | bootstrap: 98 recusados, 748 aprovados
        │                                               │       │       └─ Não: NÓ 1429 | property_value (valor do imóvel em USD) <= 825000.0? | bootstrap: 25 recusados, 62 aprovados
        │                                               │       │           ├─ Sim: FOLHA 1430 | previsão: APROVADO (1) | bootstrap: 8 recusados, 10 aprovados
        │                                               │       │           └─ Não: FOLHA 1431 | previsão: APROVADO (1) | bootstrap: 17 recusados, 52 aprovados
        │                                               │       └─ Não: FOLHA 1423 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 3 aprovados
        │                                               └─ Não: NÓ 1390 | income (renda anual declarada em milhares de USD) <= 44.5? | bootstrap: 6772 recusados, 164770 aprovados
        │                                                   ├─ Sim: NÓ 1391 | lei = 'RVDPPPGHCGZ40J4VQ731'? | bootstrap: 677 recusados, 6696 aprovados
        │                                                   │   ├─ Sim: FOLHA 1401 | previsão: APROVADO (1) | bootstrap: 4 recusados, 14 aprovados
        │                                                   │   └─ Não: NÓ 1392 | lei = '549300NFPC3Y4W3Y7R87'? | bootstrap: 673 recusados, 6682 aprovados
        │                                                   │       ├─ Sim: FOLHA 1400 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
        │                                                   │       └─ Não: NÓ 1393 | applicant_age = '35-44'? | bootstrap: 668 recusados, 6677 aprovados
        │                                                   │           ├─ Sim: NÓ 1397 | lei = '549300AQ3T62GXDU7D76'? | bootstrap: 158 recusados, 1135 aprovados
        │                                                   │           │   ├─ Sim: FOLHA 1399 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
        │                                                   │           │   └─ Não: FOLHA 1398 | previsão: APROVADO (1) | bootstrap: 158 recusados, 1121 aprovados
        │                                                   │           └─ Não: NÓ 1394 | lei = '549300U6DW7DX671T306'? | bootstrap: 510 recusados, 5542 aprovados
        │                                                   │               ├─ Sim: FOLHA 1396 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
        │                                                   │               └─ Não: FOLHA 1395 | previsão: APROVADO (1) | bootstrap: 510 recusados, 5530 aprovados
        │                                                   └─ Não: NÓ 1402 | lei = '5493000NYUJT9UC6G261'? | bootstrap: 6095 recusados, 158074 aprovados
        │                                                       ├─ Sim: NÓ 1414 | loan_amount (valor solicitado em USD) <= 600000.0? | bootstrap: 19 recusados, 222 aprovados
        │                                                       │   ├─ Sim: NÓ 1415 | applicant_age = '<25'? | bootstrap: 16 recusados, 214 aprovados
        │                                                       │   │   ├─ Sim: FOLHA 1419 | previsão: APROVADO (1) | bootstrap: 1 recusados, 20 aprovados
        │                                                       │   │   └─ Não: NÓ 1416 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 15 recusados, 194 aprovados
        │                                                       │   │       ├─ Sim: FOLHA 1418 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
        │                                                       │   │       └─ Não: FOLHA 1417 | previsão: APROVADO (1) | bootstrap: 14 recusados, 180 aprovados
        │                                                       │   └─ Não: FOLHA 1420 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
        │                                                       └─ Não: NÓ 1403 | loan_amount (valor solicitado em USD) <= 180000.0? | bootstrap: 6076 recusados, 157852 aprovados
        │                                                           ├─ Sim: NÓ 1404 | lei = 'B4TYDEB6GKMZO031MB27'? | bootstrap: 1982 recusados, 29192 aprovados
        │                                                           │   ├─ Sim: FOLHA 1408 | previsão: APROVADO (1) | bootstrap: 76 recusados, 222 aprovados
        │                                                           │   └─ Não: NÓ 1405 | loan_term (prazo em meses) <= 368.5? | bootstrap: 1906 recusados, 28970 aprovados
        │                                                           │       ├─ Sim: FOLHA 1406 | previsão: APROVADO (1) | bootstrap: 1880 recusados, 28877 aprovados
        │                                                           │       └─ Não: FOLHA 1407 | previsão: APROVADO (1) | bootstrap: 26 recusados, 93 aprovados
        │                                                           └─ Não: NÓ 1409 | lei = '549300K7224BC1IAX207'? | bootstrap: 4094 recusados, 128660 aprovados
        │                                                               ├─ Sim: FOLHA 1413 | previsão: APROVADO (1) | bootstrap: 10 recusados, 39 aprovados
        │                                                               └─ Não: NÓ 1410 | lei = '549300JYXTZDSPJEPI44'? | bootstrap: 4084 recusados, 128621 aprovados
        │                                                                   ├─ Sim: FOLHA 1412 | previsão: APROVADO (1) | bootstrap: 66 recusados, 376 aprovados
        │                                                                   └─ Não: FOLHA 1411 | previsão: APROVADO (1) | bootstrap: 4018 recusados, 128245 aprovados
        └─ Não: NÓ 3 | debt_to_income_ratio = '43'? | bootstrap: 152578 recusados, 1080183 aprovados
            ├─ Sim: NÓ 1215 | lei = '5493000YNV8IX4VD3X12'? | bootstrap: 3755 recusados, 54382 aprovados
            │   ├─ Sim: NÓ 1367 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 401 recusados, 262 aprovados
            │   │   ├─ Sim: NÓ 1373 | loan_term (prazo em meses) <= 330.0? | bootstrap: 358 recusados, 234 aprovados
            │   │   │   ├─ Sim: FOLHA 1374 | previsão: RECUSADO (0) | bootstrap: 321 recusados, 144 aprovados
            │   │   │   └─ Não: NÓ 1375 | applicant_age = '55-64'? | bootstrap: 37 recusados, 90 aprovados
            │   │   │       ├─ Sim: FOLHA 1377 | previsão: APROVADO (1) | bootstrap: 5 recusados, 14 aprovados
            │   │   │       └─ Não: FOLHA 1376 | previsão: APROVADO (1) | bootstrap: 32 recusados, 76 aprovados
            │   │   └─ Não: NÓ 1368 | derived_ethnicity = 'Joint'? | bootstrap: 43 recusados, 28 aprovados
            │   │       ├─ Sim: FOLHA 1372 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 5 aprovados
            │   │       └─ Não: NÓ 1369 | construction_method = '1'? | bootstrap: 35 recusados, 23 aprovados
            │   │           ├─ Sim: FOLHA 1371 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
            │   │           └─ Não: FOLHA 1370 | previsão: RECUSADO (0) | bootstrap: 34 recusados, 17 aprovados
            │   └─ Não: NÓ 1216 | lei = '549300XQVJ1XBNFA5536'? | bootstrap: 3354 recusados, 54120 aprovados
            │       ├─ Sim: NÓ 1284 | income (renda anual declarada em milhares de USD) <= 106.0? | bootstrap: 202 recusados, 431 aprovados
            │       │   ├─ Sim: NÓ 1285 | derived_race = 'Black or African American'? | bootstrap: 175 recusados, 398 aprovados
            │       │   │   ├─ Sim: FOLHA 1357 | previsão: APROVADO (1) | bootstrap: 33 recusados, 55 aprovados
            │       │   │   └─ Não: NÓ 1286 | derived_ethnicity = 'Joint'? | bootstrap: 142 recusados, 343 aprovados
            │       │   │       ├─ Sim: FOLHA 1356 | previsão: APROVADO (1) | bootstrap: 5 recusados, 22 aprovados
            │       │   │       └─ Não: NÓ 1287 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 137 recusados, 321 aprovados
            │       │   │           ├─ Sim: FOLHA 1355 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
            │       │   │           └─ Não: NÓ 1288 | applicant_age = '<25'? | bootstrap: 133 recusados, 316 aprovados
            │       │   │               ├─ Sim: NÓ 1350 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 27 recusados, 43 aprovados
            │       │   │               │   ├─ Sim: NÓ 1352 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 103.0? | bootstrap: 10 recusados, 10 aprovados
            │       │   │               │   │   ├─ Sim: FOLHA 1353 | previsão: APROVADO (1) | bootstrap: 5 recusados, 9 aprovados
            │       │   │               │   │   └─ Não: FOLHA 1354 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 1 aprovados
            │       │   │               │   └─ Não: FOLHA 1351 | previsão: APROVADO (1) | bootstrap: 17 recusados, 33 aprovados
            │       │   │               └─ Não: NÓ 1289 | applicant_age = '35-44'? | bootstrap: 106 recusados, 273 aprovados
            │       │   │                   ├─ Sim: NÓ 1341 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 44 recusados, 70 aprovados
            │       │   │                   │   ├─ Sim: NÓ 1343 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 95.5? | bootstrap: 12 recusados, 19 aprovados
            │       │   │                   │   │   ├─ Sim: NÓ 1344 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 84.5? | bootstrap: 6 recusados, 16 aprovados
            │       │   │                   │   │   │   ├─ Sim: NÓ 1345 | property_value (valor do imóvel em USD) <= 90000.0? | bootstrap: 6 recusados, 7 aprovados
            │       │   │                   │   │   │   │   ├─ Sim: FOLHA 1346 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 3 aprovados
            │       │   │                   │   │   │   │   └─ Não: FOLHA 1347 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
            │       │   │                   │   │   │   └─ Não: FOLHA 1348 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
            │       │   │                   │   │   └─ Não: FOLHA 1349 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 3 aprovados
            │       │   │                   │   └─ Não: FOLHA 1342 | previsão: APROVADO (1) | bootstrap: 32 recusados, 51 aprovados
            │       │   │                   └─ Não: NÓ 1290 | derived_race = 'White'? | bootstrap: 62 recusados, 203 aprovados
            │       │   │                       ├─ Sim: NÓ 1294 | applicant_age = '25-34'? | bootstrap: 57 recusados, 181 aprovados
            │       │   │                       │   ├─ Sim: NÓ 1308 | loan_term (prazo em meses) <= 288.0? | bootstrap: 49 recusados, 102 aprovados
            │       │   │                       │   │   ├─ Sim: NÓ 1309 | loan_amount (valor solicitado em USD) <= 50000.0? | bootstrap: 18 recusados, 33 aprovados
            │       │   │                       │   │   │   ├─ Sim: NÓ 1310 | income (renda anual declarada em milhares de USD) <= 43.5? | bootstrap: 4 recusados, 16 aprovados
            │       │   │                       │   │   │   │   ├─ Sim: FOLHA 1311 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
            │       │   │                       │   │   │   │   └─ Não: NÓ 1312 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 91.0? | bootstrap: 4 recusados, 8 aprovados
            │       │   │                       │   │   │   │       ├─ Sim: FOLHA 1313 | previsão: APROVADO (1) | bootstrap: 0 recusados, 6 aprovados
            │       │   │                       │   │   │   │       └─ Não: FOLHA 1314 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 2 aprovados
            │       │   │                       │   │   │   └─ Não: NÓ 1315 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 90.0? | bootstrap: 14 recusados, 17 aprovados
            │       │   │                       │   │   │       ├─ Sim: NÓ 1316 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 72.5? | bootstrap: 3 recusados, 10 aprovados
            │       │   │                       │   │   │       │   ├─ Sim: FOLHA 1317 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
            │       │   │                       │   │   │       │   └─ Não: FOLHA 1318 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
            │       │   │                       │   │   │       └─ Não: NÓ 1319 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 112.5? | bootstrap: 11 recusados, 7 aprovados
            │       │   │                       │   │   │           ├─ Sim: FOLHA 1320 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
            │       │   │                       │   │   │           └─ Não: FOLHA 1321 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 2 aprovados
            │       │   │                       │   │   └─ Não: NÓ 1322 | income (renda anual declarada em milhares de USD) <= 64.0? | bootstrap: 31 recusados, 69 aprovados
            │       │   │                       │   │       ├─ Sim: NÓ 1323 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 67.5? | bootstrap: 21 recusados, 33 aprovados
            │       │   │                       │   │       │   ├─ Sim: FOLHA 1324 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
            │       │   │                       │   │       │   └─ Não: NÓ 1325 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 20 recusados, 19 aprovados
            │       │   │                       │   │       │       ├─ Sim: FOLHA 1331 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
            │       │   │                       │   │       │       └─ Não: NÓ 1326 | loan_amount (valor solicitado em USD) <= 80000.0? | bootstrap: 16 recusados, 14 aprovados
            │       │   │                       │   │       │           ├─ Sim: FOLHA 1327 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 3 aprovados
            │       │   │                       │   │       │           └─ Não: NÓ 1328 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 108.0? | bootstrap: 7 recusados, 11 aprovados
            │       │   │                       │   │       │               ├─ Sim: FOLHA 1329 | previsão: APROVADO (1) | bootstrap: 4 recusados, 7 aprovados
            │       │   │                       │   │       │               └─ Não: FOLHA 1330 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
            │       │   │                       │   │       └─ Não: NÓ 1332 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 10 recusados, 36 aprovados
            │       │   │                       │   │           ├─ Sim: NÓ 1334 | property_value (valor do imóvel em USD) <= 110000.0? | bootstrap: 9 recusados, 26 aprovados
            │       │   │                       │   │           │   ├─ Sim: FOLHA 1335 | previsão: APROVADO (1) | bootstrap: 1 recusados, 10 aprovados
            │       │   │                       │   │           │   └─ Não: NÓ 1336 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 95.5? | bootstrap: 8 recusados, 16 aprovados
            │       │   │                       │   │           │       ├─ Sim: FOLHA 1337 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
            │       │   │                       │   │           │       └─ Não: NÓ 1338 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 117.5? | bootstrap: 7 recusados, 10 aprovados
            │       │   │                       │   │           │           ├─ Sim: FOLHA 1339 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
            │       │   │                       │   │           │           └─ Não: FOLHA 1340 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
            │       │   │                       │   │           └─ Não: FOLHA 1333 | previsão: APROVADO (1) | bootstrap: 1 recusados, 10 aprovados
            │       │   │                       │   └─ Não: NÓ 1295 | applicant_age = '55-64'? | bootstrap: 8 recusados, 79 aprovados
            │       │   │                       │       ├─ Sim: NÓ 1299 | loan_term (prazo em meses) <= 280.5? | bootstrap: 7 recusados, 42 aprovados
            │       │   │                       │       │   ├─ Sim: NÓ 1300 | income (renda anual declarada em milhares de USD) <= 57.0? | bootstrap: 5 recusados, 14 aprovados
            │       │   │                       │       │   │   ├─ Sim: FOLHA 1301 | previsão: APROVADO (1) | bootstrap: 1 recusados, 11 aprovados
            │       │   │                       │       │   │   └─ Não: FOLHA 1302 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 3 aprovados
            │       │   │                       │       │   └─ Não: NÓ 1303 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 2 recusados, 28 aprovados
            │       │   │                       │       │       ├─ Sim: FOLHA 1307 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
            │       │   │                       │       │       └─ Não: NÓ 1304 | loan_amount (valor solicitado em USD) <= 135000.0? | bootstrap: 2 recusados, 17 aprovados
            │       │   │                       │       │           ├─ Sim: FOLHA 1305 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
            │       │   │                       │       │           └─ Não: FOLHA 1306 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
            │       │   │                       │       └─ Não: NÓ 1296 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 115.0? | bootstrap: 1 recusados, 37 aprovados
            │       │   │                       │           ├─ Sim: FOLHA 1297 | previsão: APROVADO (1) | bootstrap: 0 recusados, 30 aprovados
            │       │   │                       │           └─ Não: FOLHA 1298 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
            │       │   │                       └─ Não: NÓ 1291 | income (renda anual declarada em milhares de USD) <= 82.5? | bootstrap: 5 recusados, 22 aprovados
            │       │   │                           ├─ Sim: FOLHA 1292 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
            │       │   │                           └─ Não: FOLHA 1293 | previsão: APROVADO (1) | bootstrap: 3 recusados, 10 aprovados
            │       │   └─ Não: NÓ 1358 | derived_race = 'White'? | bootstrap: 27 recusados, 33 aprovados
            │       │       ├─ Sim: NÓ 1360 | applicant_age = '25-34'? | bootstrap: 22 recusados, 27 aprovados
            │       │       │   ├─ Sim: FOLHA 1366 | previsão: APROVADO (1) | bootstrap: 8 recusados, 9 aprovados
            │       │       │   └─ Não: NÓ 1361 | applicant_age = '35-44'? | bootstrap: 14 recusados, 18 aprovados
            │       │       │       ├─ Sim: NÓ 1363 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 110.5? | bootstrap: 8 recusados, 10 aprovados
            │       │       │       │   ├─ Sim: FOLHA 1364 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
            │       │       │       │   └─ Não: FOLHA 1365 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 5 aprovados
            │       │       │       └─ Não: FOLHA 1362 | previsão: APROVADO (1) | bootstrap: 6 recusados, 8 aprovados
            │       │       └─ Não: FOLHA 1359 | previsão: APROVADO (1) | bootstrap: 5 recusados, 6 aprovados
            │       └─ Não: NÓ 1217 | lei = '549300MZ8VZJOVC63092'? | bootstrap: 3152 recusados, 53689 aprovados
            │           ├─ Sim: NÓ 1275 | income (renda anual declarada em milhares de USD) <= 294.5? | bootstrap: 58 recusados, 255 aprovados
            │           │   ├─ Sim: NÓ 1276 | applicant_age = '>74'? | bootstrap: 52 recusados, 251 aprovados
            │           │   │   ├─ Sim: FOLHA 1282 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
            │           │   │   └─ Não: NÓ 1277 | property_value (valor do imóvel em USD) <= 260000.0? | bootstrap: 52 recusados, 243 aprovados
            │           │   │       ├─ Sim: NÓ 1278 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 109.5? | bootstrap: 11 recusados, 25 aprovados
            │           │   │       │   ├─ Sim: FOLHA 1279 | previsão: APROVADO (1) | bootstrap: 11 recusados, 18 aprovados
            │           │   │       │   └─ Não: FOLHA 1280 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
            │           │   │       └─ Não: FOLHA 1281 | previsão: APROVADO (1) | bootstrap: 41 recusados, 218 aprovados
            │           │   └─ Não: FOLHA 1283 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 4 aprovados
            │           └─ Não: NÓ 1218 | lei = '549300VHBALPLHRR4080'? | bootstrap: 3094 recusados, 53434 aprovados
            │               ├─ Sim: FOLHA 1274 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
            │               └─ Não: NÓ 1219 | lei = '549300HTRLFU2IIL4380'? | bootstrap: 3092 recusados, 53428 aprovados
            │                   ├─ Sim: FOLHA 1273 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
            │                   └─ Não: NÓ 1220 | lei = '549300VZVN841I2ILS84'? | bootstrap: 3089 recusados, 53416 aprovados
            │                       ├─ Sim: NÓ 1266 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 65.5? | bootstrap: 27 recusados, 1369 aprovados
            │                       │   ├─ Sim: NÓ 1267 | property_value (valor do imóvel em USD) <= 170000.0? | bootstrap: 7 recusados, 124 aprovados
            │                       │   │   ├─ Sim: NÓ 1268 | loan_type = '1'? | bootstrap: 3 recusados, 24 aprovados
            │                       │   │   │   ├─ Sim: FOLHA 1270 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
            │                       │   │   │   └─ Não: FOLHA 1269 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
            │                       │   │   └─ Não: FOLHA 1271 | previsão: APROVADO (1) | bootstrap: 4 recusados, 100 aprovados
            │                       │   └─ Não: FOLHA 1272 | previsão: APROVADO (1) | bootstrap: 20 recusados, 1245 aprovados
            │                       └─ Não: NÓ 1221 | lei = '549300U3721PJGQZYY68'? | bootstrap: 3062 recusados, 52047 aprovados
            │                           ├─ Sim: NÓ 1253 | loan_type = '1'? | bootstrap: 18 recusados, 954 aprovados
            │                           │   ├─ Sim: FOLHA 1265 | previsão: APROVADO (1) | bootstrap: 14 recusados, 716 aprovados
            │                           │   └─ Não: NÓ 1254 | income (renda anual declarada em milhares de USD) <= 60.5? | bootstrap: 4 recusados, 238 aprovados
            │                           │       ├─ Sim: NÓ 1255 | loan_type = '3'? | bootstrap: 2 recusados, 49 aprovados
            │                           │       │   ├─ Sim: FOLHA 1259 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
            │                           │       │   └─ Não: NÓ 1256 | income (renda anual declarada em milhares de USD) <= 58.5? | bootstrap: 2 recusados, 35 aprovados
            │                           │       │       ├─ Sim: FOLHA 1257 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
            │                           │       │       └─ Não: FOLHA 1258 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
            │                           │       └─ Não: NÓ 1260 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 2 recusados, 189 aprovados
            │                           │           ├─ Sim: FOLHA 1264 | previsão: APROVADO (1) | bootstrap: 1 recusados, 130 aprovados
            │                           │           └─ Não: NÓ 1261 | applicant_age = '25-34'? | bootstrap: 1 recusados, 59 aprovados
            │                           │               ├─ Sim: FOLHA 1263 | previsão: APROVADO (1) | bootstrap: 1 recusados, 24 aprovados
            │                           │               └─ Não: FOLHA 1262 | previsão: APROVADO (1) | bootstrap: 0 recusados, 35 aprovados
            │                           └─ Não: NÓ 1222 | lei = '549300GBZNWRGLS7CP89'? | bootstrap: 3044 recusados, 51093 aprovados
            │                               ├─ Sim: FOLHA 1252 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
            │                               └─ Não: NÓ 1223 | lei = '5493009XSE00M8B9R434'? | bootstrap: 3041 recusados, 51088 aprovados
            │                                   ├─ Sim: NÓ 1249 | loan_term (prazo em meses) <= 288.0? | bootstrap: 6 recusados, 10 aprovados
            │                                   │   ├─ Sim: FOLHA 1250 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
            │                                   │   └─ Não: FOLHA 1251 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 3 aprovados
            │                                   └─ Não: NÓ 1224 | lei = 'X05BVSK68TQ7YTOSNR22'? | bootstrap: 3035 recusados, 51078 aprovados
            │                                       ├─ Sim: FOLHA 1248 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 0 aprovados
            │                                       └─ Não: NÓ 1225 | lei = '549300K7224BC1IAX207'? | bootstrap: 3028 recusados, 51078 aprovados
            │                                           ├─ Sim: FOLHA 1247 | previsão: APROVADO (1) | bootstrap: 12 recusados, 15 aprovados
            │                                           └─ Não: NÓ 1226 | construction_method = '1'? | bootstrap: 3016 recusados, 51063 aprovados
            │                                               ├─ Sim: NÓ 1228 | lei = '549300FX7K8PTEQUU487'? | bootstrap: 2604 recusados, 49201 aprovados
            │                                               │   ├─ Sim: NÓ 1240 | applicant_age = '55-64'? | bootstrap: 23 recusados, 56 aprovados
            │                                               │   │   ├─ Sim: FOLHA 1246 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
            │                                               │   │   └─ Não: NÓ 1241 | property_value (valor do imóvel em USD) <= 530000.0? | bootstrap: 18 recusados, 52 aprovados
            │                                               │   │       ├─ Sim: FOLHA 1242 | previsão: APROVADO (1) | bootstrap: 4 recusados, 30 aprovados
            │                                               │   │       └─ Não: NÓ 1243 | applicant_age = '35-44'? | bootstrap: 14 recusados, 22 aprovados
            │                                               │   │           ├─ Sim: FOLHA 1245 | previsão: APROVADO (1) | bootstrap: 9 recusados, 17 aprovados
            │                                               │   │           └─ Não: FOLHA 1244 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
            │                                               │   └─ Não: NÓ 1229 | lei = 'RVDPPPGHCGZ40J4VQ731'? | bootstrap: 2581 recusados, 49145 aprovados
            │                                               │       ├─ Sim: NÓ 1235 | loan_type = '2'? | bootstrap: 39 recusados, 330 aprovados
            │                                               │       │   ├─ Sim: FOLHA 1239 | previsão: APROVADO (1) | bootstrap: 6 recusados, 84 aprovados
            │                                               │       │   └─ Não: NÓ 1236 | loan_type = '3'? | bootstrap: 33 recusados, 246 aprovados
            │                                               │       │       ├─ Sim: FOLHA 1238 | previsão: APROVADO (1) | bootstrap: 2 recusados, 28 aprovados
            │                                               │       │       └─ Não: FOLHA 1237 | previsão: APROVADO (1) | bootstrap: 31 recusados, 218 aprovados
            │                                               │       └─ Não: NÓ 1230 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 2542 recusados, 48815 aprovados
            │                                               │           ├─ Sim: FOLHA 1234 | previsão: APROVADO (1) | bootstrap: 16 recusados, 99 aprovados
            │                                               │           └─ Não: NÓ 1231 | lei = 'B4TYDEB6GKMZO031MB27'? | bootstrap: 2526 recusados, 48716 aprovados
            │                                               │               ├─ Sim: FOLHA 1233 | previsão: APROVADO (1) | bootstrap: 59 recusados, 483 aprovados
            │                                               │               └─ Não: FOLHA 1232 | previsão: APROVADO (1) | bootstrap: 2467 recusados, 48233 aprovados
            │                                               └─ Não: FOLHA 1227 | previsão: APROVADO (1) | bootstrap: 412 recusados, 1862 aprovados
            └─ Não: NÓ 4 | derived_race = 'White'? | bootstrap: 148823 recusados, 1025801 aprovados
                ├─ Sim: NÓ 448 | lei = '549300CB67L6KPJLHE19'? | bootstrap: 103964 recusados, 796509 aprovados
                │   ├─ Sim: NÓ 960 | debt_to_income_ratio = '44'? | bootstrap: 5493 recusados, 5504 aprovados
                │   │   ├─ Sim: NÓ 1196 | applicant_age = '25-34'? | bootstrap: 59 recusados, 347 aprovados
                │   │   │   ├─ Sim: FOLHA 1214 | previsão: APROVADO (1) | bootstrap: 16 recusados, 107 aprovados
                │   │   │   └─ Não: NÓ 1197 | applicant_age = '>74'? | bootstrap: 43 recusados, 240 aprovados
                │   │   │       ├─ Sim: NÓ 1211 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 85.0? | bootstrap: 2 recusados, 23 aprovados
                │   │   │       │   ├─ Sim: FOLHA 1212 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
                │   │   │       │   └─ Não: FOLHA 1213 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                │   │   │       └─ Não: NÓ 1198 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 41 recusados, 217 aprovados
                │   │   │           ├─ Sim: NÓ 1200 | income (renda anual declarada em milhares de USD) <= 80.5? | bootstrap: 8 recusados, 57 aprovados
                │   │   │           │   ├─ Sim: NÓ 1201 | income (renda anual declarada em milhares de USD) <= 42.5? | bootstrap: 3 recusados, 32 aprovados
                │   │   │           │   │   ├─ Sim: FOLHA 1202 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
                │   │   │           │   │   └─ Não: NÓ 1203 | applicant_age = '35-44'? | bootstrap: 2 recusados, 28 aprovados
                │   │   │           │   │       ├─ Sim: FOLHA 1205 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
                │   │   │           │   │       └─ Não: FOLHA 1204 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
                │   │   │           │   └─ Não: NÓ 1206 | income (renda anual declarada em milhares de USD) <= 93.0? | bootstrap: 5 recusados, 25 aprovados
                │   │   │           │       ├─ Sim: FOLHA 1207 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │   │   │           │       └─ Não: NÓ 1208 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 84.5? | bootstrap: 2 recusados, 21 aprovados
                │   │   │           │           ├─ Sim: FOLHA 1209 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
                │   │   │           │           └─ Não: FOLHA 1210 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
                │   │   │           └─ Não: FOLHA 1199 | previsão: APROVADO (1) | bootstrap: 33 recusados, 160 aprovados
                │   │   └─ Não: NÓ 961 | debt_to_income_ratio = '47'? | bootstrap: 5434 recusados, 5157 aprovados
                │   │       ├─ Sim: NÓ 1109 | applicant_age = '25-34'? | bootstrap: 84 recusados, 723 aprovados
                │   │       │   ├─ Sim: NÓ 1155 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 31 recusados, 214 aprovados
                │   │       │   │   ├─ Sim: NÓ 1163 | property_value (valor do imóvel em USD) <= 110000.0? | bootstrap: 24 recusados, 156 aprovados
                │   │       │   │   │   ├─ Sim: NÓ 1164 | loan_term (prazo em meses) <= 270.0? | bootstrap: 2 recusados, 54 aprovados
                │   │       │   │   │   │   ├─ Sim: FOLHA 1165 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                │   │       │   │   │   │   └─ Não: NÓ 1166 | income (renda anual declarada em milhares de USD) <= 64.5? | bootstrap: 2 recusados, 45 aprovados
                │   │       │   │   │   │       ├─ Sim: NÓ 1167 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 92.5? | bootstrap: 1 recusados, 38 aprovados
                │   │       │   │   │   │       │   ├─ Sim: NÓ 1168 | income (renda anual declarada em milhares de USD) <= 42.0? | bootstrap: 1 recusados, 16 aprovados
                │   │       │   │   │   │       │   │   ├─ Sim: FOLHA 1169 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                │   │       │   │   │   │       │   │   └─ Não: FOLHA 1170 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │   │       │   │   │   │       │   └─ Não: FOLHA 1171 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
                │   │       │   │   │   │       └─ Não: FOLHA 1172 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
                │   │       │   │   │   └─ Não: NÓ 1173 | loan_term (prazo em meses) <= 330.0? | bootstrap: 22 recusados, 102 aprovados
                │   │       │   │   │       ├─ Sim: NÓ 1174 | loan_amount (valor solicitado em USD) <= 190000.0? | bootstrap: 19 recusados, 84 aprovados
                │   │       │   │   │       │   ├─ Sim: NÓ 1175 | income (renda anual declarada em milhares de USD) <= 71.5? | bootstrap: 14 recusados, 69 aprovados
                │   │       │   │   │       │   │   ├─ Sim: NÓ 1176 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 96.5? | bootstrap: 6 recusados, 54 aprovados
                │   │       │   │   │       │   │   │   ├─ Sim: NÓ 1177 | loan_amount (valor solicitado em USD) <= 120000.0? | bootstrap: 4 recusados, 20 aprovados
                │   │       │   │   │       │   │   │   │   ├─ Sim: FOLHA 1178 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                │   │       │   │   │       │   │   │   │   └─ Não: FOLHA 1179 | previsão: APROVADO (1) | bootstrap: 4 recusados, 11 aprovados
                │   │       │   │   │       │   │   │   └─ Não: NÓ 1180 | loan_amount (valor solicitado em USD) <= 120000.0? | bootstrap: 2 recusados, 34 aprovados
                │   │       │   │   │       │   │   │       ├─ Sim: FOLHA 1181 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
                │   │       │   │   │       │   │   │       └─ Não: NÓ 1182 | loan_amount (valor solicitado em USD) <= 140000.0? | bootstrap: 1 recusados, 27 aprovados
                │   │       │   │   │       │   │   │           ├─ Sim: FOLHA 1183 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
                │   │       │   │   │       │   │   │           └─ Não: FOLHA 1184 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
                │   │       │   │   │       │   │   └─ Não: NÓ 1185 | property_value (valor do imóvel em USD) <= 170000.0? | bootstrap: 8 recusados, 15 aprovados
                │   │       │   │   │       │   │       ├─ Sim: FOLHA 1186 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 8 aprovados
                │   │       │   │   │       │   │       └─ Não: FOLHA 1187 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                │   │       │   │   │       │   └─ Não: NÓ 1188 | property_value (valor do imóvel em USD) <= 230000.0? | bootstrap: 5 recusados, 15 aprovados
                │   │       │   │   │       │       ├─ Sim: FOLHA 1189 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
                │   │       │   │   │       │       └─ Não: FOLHA 1190 | previsão: APROVADO (1) | bootstrap: 1 recusados, 10 aprovados
                │   │       │   │   │       └─ Não: NÓ 1191 | loan_amount (valor solicitado em USD) <= 290000.0? | bootstrap: 3 recusados, 18 aprovados
                │   │       │   │   │           ├─ Sim: NÓ 1192 | property_value (valor do imóvel em USD) <= 210000.0? | bootstrap: 1 recusados, 14 aprovados
                │   │       │   │   │           │   ├─ Sim: FOLHA 1193 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
                │   │       │   │   │           │   └─ Não: FOLHA 1194 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │   │       │   │   │           └─ Não: FOLHA 1195 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                │   │       │   │   └─ Não: NÓ 1156 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 7 recusados, 58 aprovados
                │   │       │   │       ├─ Sim: NÓ 1158 | property_value (valor do imóvel em USD) <= 80000.0? | bootstrap: 7 recusados, 52 aprovados
                │   │       │   │       │   ├─ Sim: FOLHA 1159 | previsão: APROVADO (1) | bootstrap: 4 recusados, 8 aprovados
                │   │       │   │       │   └─ Não: NÓ 1160 | property_value (valor do imóvel em USD) <= 190000.0? | bootstrap: 3 recusados, 44 aprovados
                │   │       │   │       │       ├─ Sim: FOLHA 1161 | previsão: APROVADO (1) | bootstrap: 1 recusados, 29 aprovados
                │   │       │   │       │       └─ Não: FOLHA 1162 | previsão: APROVADO (1) | bootstrap: 2 recusados, 15 aprovados
                │   │       │   │       └─ Não: FOLHA 1157 | previsão: APROVADO (1) | bootstrap: 0 recusados, 6 aprovados
                │   │       │   └─ Não: NÓ 1110 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 53 recusados, 509 aprovados
                │   │       │       ├─ Sim: NÓ 1118 | property_value (valor do imóvel em USD) <= 130000.0? | bootstrap: 40 recusados, 397 aprovados
                │   │       │       │   ├─ Sim: NÓ 1119 | loan_amount (valor solicitado em USD) <= 120000.0? | bootstrap: 29 recusados, 195 aprovados
                │   │       │       │   │   ├─ Sim: NÓ 1120 | loan_term (prazo em meses) <= 270.0? | bootstrap: 19 recusados, 190 aprovados
                │   │       │       │   │   │   ├─ Sim: FOLHA 1121 | previsão: APROVADO (1) | bootstrap: 3 recusados, 23 aprovados
                │   │       │       │   │   │   └─ Não: NÓ 1122 | applicant_age = '55-64'? | bootstrap: 16 recusados, 167 aprovados
                │   │       │       │   │   │       ├─ Sim: NÓ 1134 | loan_amount (valor solicitado em USD) <= 70000.0? | bootstrap: 5 recusados, 35 aprovados
                │   │       │       │   │   │       │   ├─ Sim: FOLHA 1135 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
                │   │       │       │   │   │       │   └─ Não: NÓ 1136 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 108.5? | bootstrap: 2 recusados, 27 aprovados
                │   │       │       │   │   │       │       ├─ Sim: FOLHA 1137 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
                │   │       │       │   │   │       │       └─ Não: FOLHA 1138 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
                │   │       │       │   │   │       └─ Não: NÓ 1123 | loan_amount (valor solicitado em USD) <= 70000.0? | bootstrap: 11 recusados, 132 aprovados
                │   │       │       │   │   │           ├─ Sim: NÓ 1124 | applicant_age = '35-44'? | bootstrap: 1 recusados, 33 aprovados
                │   │       │       │   │   │           │   ├─ Sim: FOLHA 1126 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │   │       │       │   │   │           │   └─ Não: FOLHA 1125 | previsão: APROVADO (1) | bootstrap: 1 recusados, 23 aprovados
                │   │       │       │   │   │           └─ Não: NÓ 1127 | income (renda anual declarada em milhares de USD) <= 65.0? | bootstrap: 10 recusados, 99 aprovados
                │   │       │       │   │   │               ├─ Sim: NÓ 1128 | loan_amount (valor solicitado em USD) <= 80000.0? | bootstrap: 6 recusados, 84 aprovados
                │   │       │       │   │   │               │   ├─ Sim: FOLHA 1129 | previsão: APROVADO (1) | bootstrap: 3 recusados, 12 aprovados
                │   │       │       │   │   │               │   └─ Não: FOLHA 1130 | previsão: APROVADO (1) | bootstrap: 3 recusados, 72 aprovados
                │   │       │       │   │   │               └─ Não: NÓ 1131 | applicant_age = '35-44'? | bootstrap: 4 recusados, 15 aprovados
                │   │       │       │   │   │                   ├─ Sim: FOLHA 1133 | previsão: RECUSADO (0) | bootstrap: 3 recusados, 3 aprovados
                │   │       │       │   │   │                   └─ Não: FOLHA 1132 | previsão: APROVADO (1) | bootstrap: 1 recusados, 12 aprovados
                │   │       │       │   │   └─ Não: FOLHA 1139 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 5 aprovados
                │   │       │       │   └─ Não: NÓ 1140 | property_value (valor do imóvel em USD) <= 290000.0? | bootstrap: 11 recusados, 202 aprovados
                │   │       │       │       ├─ Sim: NÓ 1141 | loan_term (prazo em meses) <= 369.0? | bootstrap: 11 recusados, 175 aprovados
                │   │       │       │       │   ├─ Sim: NÓ 1142 | applicant_age = '<25'? | bootstrap: 9 recusados, 159 aprovados
                │   │       │       │       │   │   ├─ Sim: NÓ 1150 | property_value (valor do imóvel em USD) <= 160000.0? | bootstrap: 3 recusados, 20 aprovados
                │   │       │       │       │   │   │   ├─ Sim: FOLHA 1151 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                │   │       │       │       │   │   │   └─ Não: FOLHA 1152 | previsão: APROVADO (1) | bootstrap: 3 recusados, 11 aprovados
                │   │       │       │       │   │   └─ Não: NÓ 1143 | applicant_age = '65-74'? | bootstrap: 6 recusados, 139 aprovados
                │   │       │       │       │   │       ├─ Sim: FOLHA 1149 | previsão: APROVADO (1) | bootstrap: 3 recusados, 33 aprovados
                │   │       │       │       │   │       └─ Não: NÓ 1144 | income (renda anual declarada em milhares de USD) <= 71.5? | bootstrap: 3 recusados, 106 aprovados
                │   │       │       │       │   │           ├─ Sim: FOLHA 1145 | previsão: APROVADO (1) | bootstrap: 0 recusados, 64 aprovados
                │   │       │       │       │   │           └─ Não: NÓ 1146 | applicant_age = '55-64'? | bootstrap: 3 recusados, 42 aprovados
                │   │       │       │       │   │               ├─ Sim: FOLHA 1148 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │   │       │       │       │   │               └─ Não: FOLHA 1147 | previsão: APROVADO (1) | bootstrap: 3 recusados, 28 aprovados
                │   │       │       │       │   └─ Não: FOLHA 1153 | previsão: APROVADO (1) | bootstrap: 2 recusados, 16 aprovados
                │   │       │       │       └─ Não: FOLHA 1154 | previsão: APROVADO (1) | bootstrap: 0 recusados, 27 aprovados
                │   │       │       └─ Não: NÓ 1111 | property_value (valor do imóvel em USD) <= 110000.0? | bootstrap: 13 recusados, 112 aprovados
                │   │       │           ├─ Sim: NÓ 1112 | loan_amount (valor solicitado em USD) <= 90000.0? | bootstrap: 7 recusados, 32 aprovados
                │   │       │           │   ├─ Sim: FOLHA 1113 | previsão: APROVADO (1) | bootstrap: 4 recusados, 25 aprovados
                │   │       │           │   └─ Não: FOLHA 1114 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
                │   │       │           └─ Não: NÓ 1115 | loan_amount (valor solicitado em USD) <= 300000.0? | bootstrap: 6 recusados, 80 aprovados
                │   │       │               ├─ Sim: FOLHA 1116 | previsão: APROVADO (1) | bootstrap: 4 recusados, 72 aprovados
                │   │       │               └─ Não: FOLHA 1117 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
                │   │       └─ Não: NÓ 962 | loan_term (prazo em meses) <= 370.0? | bootstrap: 5350 recusados, 4434 aprovados
                │   │           ├─ Sim: NÓ 963 | debt_to_income_ratio = '>60%'? | bootstrap: 4982 recusados, 4293 aprovados
                │   │           │   ├─ Sim: NÓ 1031 | income (renda anual declarada em milhares de USD) <= 52.5? | bootstrap: 1693 recusados, 18 aprovados
                │   │           │   │   ├─ Sim: NÓ 1032 | applicant_age = '65-74'? | bootstrap: 1182 recusados, 17 aprovados
                │   │           │   │   │   ├─ Sim: NÓ 1056 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 79.5? | bootstrap: 117 recusados, 6 aprovados
                │   │           │   │   │   │   ├─ Sim: NÓ 1057 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 75.0? | bootstrap: 25 recusados, 5 aprovados
                │   │           │   │   │   │   │   ├─ Sim: FOLHA 1058 | previsão: RECUSADO (0) | bootstrap: 21 recusados, 2 aprovados
                │   │           │   │   │   │   │   └─ Não: FOLHA 1059 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 3 aprovados
                │   │           │   │   │   │   └─ Não: NÓ 1060 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 92 recusados, 1 aprovados
                │   │           │   │   │   │       ├─ Sim: NÓ 1062 | loan_term (prazo em meses) <= 270.0? | bootstrap: 85 recusados, 1 aprovados
                │   │           │   │   │   │       │   ├─ Sim: FOLHA 1063 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 0 aprovados
                │   │           │   │   │   │       │   └─ Não: NÓ 1064 | income (renda anual declarada em milhares de USD) <= 46.0? | bootstrap: 75 recusados, 1 aprovados
                │   │           │   │   │   │       │       ├─ Sim: FOLHA 1065 | previsão: RECUSADO (0) | bootstrap: 66 recusados, 0 aprovados
                │   │           │   │   │   │       │       └─ Não: FOLHA 1066 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 1 aprovados
                │   │           │   │   │   │       └─ Não: FOLHA 1061 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 0 aprovados
                │   │           │   │   │   └─ Não: NÓ 1033 | applicant_age = '>74'? | bootstrap: 1065 recusados, 11 aprovados
                │   │           │   │   │       ├─ Sim: FOLHA 1055 | previsão: RECUSADO (0) | bootstrap: 67 recusados, 0 aprovados
                │   │           │   │   │       └─ Não: NÓ 1034 | income (renda anual declarada em milhares de USD) <= 50.5? | bootstrap: 998 recusados, 11 aprovados
                │   │           │   │   │           ├─ Sim: NÓ 1035 | loan_term (prazo em meses) <= 270.0? | bootstrap: 959 recusados, 8 aprovados
                │   │           │   │   │           │   ├─ Sim: FOLHA 1036 | previsão: RECUSADO (0) | bootstrap: 43 recusados, 0 aprovados
                │   │           │   │   │           │   └─ Não: NÓ 1037 | income (renda anual declarada em milhares de USD) <= 23.5? | bootstrap: 916 recusados, 8 aprovados
                │   │           │   │   │           │       ├─ Sim: NÓ 1038 | property_value (valor do imóvel em USD) <= 80000.0? | bootstrap: 152 recusados, 4 aprovados
                │   │           │   │   │           │       │   ├─ Sim: NÓ 1039 | applicant_age = '25-34'? | bootstrap: 47 recusados, 4 aprovados
                │   │           │   │   │           │       │   │   ├─ Sim: FOLHA 1043 | previsão: RECUSADO (0) | bootstrap: 15 recusados, 0 aprovados
                │   │           │   │   │           │       │   │   └─ Não: NÓ 1040 | property_value (valor do imóvel em USD) <= 70000.0? | bootstrap: 32 recusados, 4 aprovados
                │   │           │   │   │           │       │   │       ├─ Sim: FOLHA 1041 | previsão: RECUSADO (0) | bootstrap: 23 recusados, 0 aprovados
                │   │           │   │   │           │       │   │       └─ Não: FOLHA 1042 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 4 aprovados
                │   │           │   │   │           │       │   └─ Não: FOLHA 1044 | previsão: RECUSADO (0) | bootstrap: 105 recusados, 0 aprovados
                │   │           │   │   │           │       └─ Não: NÓ 1045 | loan_term (prazo em meses) <= 330.0? | bootstrap: 764 recusados, 4 aprovados
                │   │           │   │   │           │           ├─ Sim: NÓ 1046 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 745 recusados, 4 aprovados
                │   │           │   │   │           │           │   ├─ Sim: NÓ 1048 | loan_amount (valor solicitado em USD) <= 70000.0? | bootstrap: 184 recusados, 4 aprovados
                │   │           │   │   │           │           │   │   ├─ Sim: FOLHA 1049 | previsão: RECUSADO (0) | bootstrap: 20 recusados, 2 aprovados
                │   │           │   │   │           │           │   │   └─ Não: NÓ 1050 | loan_amount (valor solicitado em USD) <= 120000.0? | bootstrap: 164 recusados, 2 aprovados
                │   │           │   │   │           │           │   │       ├─ Sim: FOLHA 1051 | previsão: RECUSADO (0) | bootstrap: 89 recusados, 0 aprovados
                │   │           │   │   │           │           │   │       └─ Não: FOLHA 1052 | previsão: RECUSADO (0) | bootstrap: 75 recusados, 2 aprovados
                │   │           │   │   │           │           │   └─ Não: FOLHA 1047 | previsão: RECUSADO (0) | bootstrap: 561 recusados, 0 aprovados
                │   │           │   │   │           │           └─ Não: FOLHA 1053 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 0 aprovados
                │   │           │   │   │           └─ Não: FOLHA 1054 | previsão: RECUSADO (0) | bootstrap: 39 recusados, 3 aprovados
                │   │           │   │   └─ Não: NÓ 1067 | applicant_age = '35-44'? | bootstrap: 511 recusados, 1 aprovados
                │   │           │   │       ├─ Sim: FOLHA 1071 | previsão: RECUSADO (0) | bootstrap: 160 recusados, 0 aprovados
                │   │           │   │       └─ Não: NÓ 1068 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 351 recusados, 1 aprovados
                │   │           │   │           ├─ Sim: FOLHA 1070 | previsão: RECUSADO (0) | bootstrap: 271 recusados, 0 aprovados
                │   │           │   │           └─ Não: FOLHA 1069 | previsão: RECUSADO (0) | bootstrap: 80 recusados, 1 aprovados
                │   │           │   └─ Não: NÓ 964 | debt_to_income_ratio = '39'? | bootstrap: 3289 recusados, 4275 aprovados
                │   │           │       ├─ Sim: NÓ 998 | loan_amount (valor solicitado em USD) <= 230000.0? | bootstrap: 75 recusados, 261 aprovados
                │   │           │       │   ├─ Sim: NÓ 999 | applicant_age = '35-44'? | bootstrap: 63 recusados, 252 aprovados
                │   │           │       │   │   ├─ Sim: FOLHA 1027 | previsão: APROVADO (1) | bootstrap: 20 recusados, 46 aprovados
                │   │           │       │   │   └─ Não: NÓ 1000 | income (renda anual declarada em milhares de USD) <= 48.0? | bootstrap: 43 recusados, 206 aprovados
                │   │           │       │   │       ├─ Sim: FOLHA 1001 | previsão: APROVADO (1) | bootstrap: 19 recusados, 32 aprovados
                │   │           │       │   │       └─ Não: NÓ 1002 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 24 recusados, 174 aprovados
                │   │           │       │   │           ├─ Sim: NÓ 1012 | property_value (valor do imóvel em USD) <= 60000.0? | bootstrap: 12 recusados, 140 aprovados
                │   │           │       │   │           │   ├─ Sim: FOLHA 1013 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                │   │           │       │   │           │   └─ Não: NÓ 1014 | income (renda anual declarada em milhares de USD) <= 69.5? | bootstrap: 10 recusados, 136 aprovados
                │   │           │       │   │           │       ├─ Sim: NÓ 1015 | loan_amount (valor solicitado em USD) <= 110000.0? | bootstrap: 1 recusados, 54 aprovados
                │   │           │       │   │           │       │   ├─ Sim: FOLHA 1016 | previsão: APROVADO (1) | bootstrap: 0 recusados, 33 aprovados
                │   │           │       │   │           │       │   └─ Não: FOLHA 1017 | previsão: APROVADO (1) | bootstrap: 1 recusados, 21 aprovados
                │   │           │       │   │           │       └─ Não: NÓ 1018 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 81.5? | bootstrap: 9 recusados, 82 aprovados
                │   │           │       │   │           │           ├─ Sim: FOLHA 1019 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
                │   │           │       │   │           │           └─ Não: NÓ 1020 | applicant_age = '25-34'? | bootstrap: 6 recusados, 73 aprovados
                │   │           │       │   │           │               ├─ Sim: NÓ 1024 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 105.5? | bootstrap: 2 recusados, 36 aprovados
                │   │           │       │   │           │               │   ├─ Sim: FOLHA 1025 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
                │   │           │       │   │           │               │   └─ Não: FOLHA 1026 | previsão: APROVADO (1) | bootstrap: 2 recusados, 14 aprovados
                │   │           │       │   │           │               └─ Não: NÓ 1021 | applicant_age = '55-64'? | bootstrap: 4 recusados, 37 aprovados
                │   │           │       │   │           │                   ├─ Sim: FOLHA 1023 | previsão: APROVADO (1) | bootstrap: 2 recusados, 15 aprovados
                │   │           │       │   │           │                   └─ Não: FOLHA 1022 | previsão: APROVADO (1) | bootstrap: 2 recusados, 22 aprovados
                │   │           │       │   │           └─ Não: NÓ 1003 | applicant_age = '<25'? | bootstrap: 12 recusados, 34 aprovados
                │   │           │       │   │               ├─ Sim: NÓ 1009 | income (renda anual declarada em milhares de USD) <= 85.5? | bootstrap: 5 recusados, 8 aprovados
                │   │           │       │   │               │   ├─ Sim: FOLHA 1010 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                │   │           │       │   │               │   └─ Não: FOLHA 1011 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │   │           │       │   │               └─ Não: NÓ 1004 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 83.0? | bootstrap: 7 recusados, 26 aprovados
                │   │           │       │   │                   ├─ Sim: FOLHA 1005 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
                │   │           │       │   │                   └─ Não: NÓ 1006 | income (renda anual declarada em milhares de USD) <= 78.5? | bootstrap: 2 recusados, 21 aprovados
                │   │           │       │   │                       ├─ Sim: FOLHA 1007 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
                │   │           │       │   │                       └─ Não: FOLHA 1008 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
                │   │           │       │   └─ Não: NÓ 1028 | property_value (valor do imóvel em USD) <= 300000.0? | bootstrap: 12 recusados, 9 aprovados
                │   │           │       │       ├─ Sim: FOLHA 1029 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 3 aprovados
                │   │           │       │       └─ Não: FOLHA 1030 | previsão: APROVADO (1) | bootstrap: 0 recusados, 6 aprovados
                │   │           │       └─ Não: NÓ 965 | applicant_age = '<25'? | bootstrap: 3214 recusados, 4014 aprovados
                │   │           │           ├─ Sim: NÓ 967 | debt_to_income_ratio = '46'? | bootstrap: 559 recusados, 520 aprovados
                │   │           │           │   ├─ Sim: NÓ 993 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 13 recusados, 71 aprovados
                │   │           │           │   │   ├─ Sim: NÓ 995 | income (renda anual declarada em milhares de USD) <= 65.0? | bootstrap: 3 recusados, 12 aprovados
                │   │           │           │   │   │   ├─ Sim: FOLHA 996 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │   │           │           │   │   │   └─ Não: FOLHA 997 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │   │           │           │   │   └─ Não: FOLHA 994 | previsão: APROVADO (1) | bootstrap: 10 recusados, 59 aprovados
                │   │           │           │   └─ Não: NÓ 968 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 546 recusados, 449 aprovados
                │   │           │           │       ├─ Sim: NÓ 976 | debt_to_income_ratio = '42'? | bootstrap: 419 recusados, 370 aprovados
                │   │           │           │       │   ├─ Sim: FOLHA 992 | previsão: APROVADO (1) | bootstrap: 4 recusados, 27 aprovados
                │   │           │           │       │   └─ Não: NÓ 977 | debt_to_income_ratio = '50%-60%'? | bootstrap: 415 recusados, 343 aprovados
                │   │           │           │       │       ├─ Sim: NÓ 989 | property_value (valor do imóvel em USD) <= 120000.0? | bootstrap: 115 recusados, 1 aprovados
                │   │           │           │       │       │   ├─ Sim: FOLHA 990 | previsão: RECUSADO (0) | bootstrap: 92 recusados, 0 aprovados
                │   │           │           │       │       │   └─ Não: FOLHA 991 | previsão: RECUSADO (0) | bootstrap: 23 recusados, 1 aprovados
                │   │           │           │       │       └─ Não: NÓ 978 | loan_term (prazo em meses) <= 270.0? | bootstrap: 300 recusados, 342 aprovados
                │   │           │           │       │           ├─ Sim: FOLHA 979 | previsão: RECUSADO (0) | bootstrap: 42 recusados, 33 aprovados
                │   │           │           │       │           └─ Não: NÓ 980 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 57.0? | bootstrap: 258 recusados, 309 aprovados
                │   │           │           │       │               ├─ Sim: FOLHA 981 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 1 aprovados
                │   │           │           │       │               └─ Não: NÓ 982 | debt_to_income_ratio = '41'? | bootstrap: 247 recusados, 308 aprovados
                │   │           │           │       │                   ├─ Sim: NÓ 986 | loan_amount (valor solicitado em USD) <= 100000.0? | bootstrap: 9 recusados, 21 aprovados
                │   │           │           │       │                   │   ├─ Sim: FOLHA 987 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
                │   │           │           │       │                   │   └─ Não: FOLHA 988 | previsão: APROVADO (1) | bootstrap: 8 recusados, 12 aprovados
                │   │           │           │       │                   └─ Não: NÓ 983 | debt_to_income_ratio = '45'? | bootstrap: 238 recusados, 287 aprovados
                │   │           │           │       │                       ├─ Sim: FOLHA 985 | previsão: APROVADO (1) | bootstrap: 3 recusados, 29 aprovados
                │   │           │           │       │                       └─ Não: FOLHA 984 | previsão: APROVADO (1) | bootstrap: 235 recusados, 258 aprovados
                │   │           │           │       └─ Não: NÓ 969 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 127 recusados, 79 aprovados
                │   │           │           │           ├─ Sim: FOLHA 975 | previsão: RECUSADO (0) | bootstrap: 107 recusados, 59 aprovados
                │   │           │           │           └─ Não: NÓ 970 | loan_amount (valor solicitado em USD) <= 90000.0? | bootstrap: 20 recusados, 20 aprovados
                │   │           │           │               ├─ Sim: FOLHA 971 | previsão: APROVADO (1) | bootstrap: 4 recusados, 13 aprovados
                │   │           │           │               └─ Não: NÓ 972 | property_value (valor do imóvel em USD) <= 140000.0? | bootstrap: 16 recusados, 7 aprovados
                │   │           │           │                   ├─ Sim: FOLHA 973 | previsão: RECUSADO (0) | bootstrap: 13 recusados, 4 aprovados
                │   │           │           │                   └─ Não: FOLHA 974 | previsão: RECUSADO (0) | bootstrap: 3 recusados, 3 aprovados
                │   │           │           └─ Não: FOLHA 966 | previsão: APROVADO (1) | bootstrap: 2655 recusados, 3494 aprovados
                │   │           └─ Não: NÓ 1072 | debt_to_income_ratio = '50%-60%'? | bootstrap: 368 recusados, 141 aprovados
                │   │               ├─ Sim: NÓ 1104 | applicant_age = '35-44'? | bootstrap: 91 recusados, 2 aprovados
                │   │               │   ├─ Sim: FOLHA 1108 | previsão: RECUSADO (0) | bootstrap: 31 recusados, 0 aprovados
                │   │               │   └─ Não: NÓ 1105 | applicant_age = '55-64'? | bootstrap: 60 recusados, 2 aprovados
                │   │               │       ├─ Sim: FOLHA 1107 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 0 aprovados
                │   │               │       └─ Não: FOLHA 1106 | previsão: RECUSADO (0) | bootstrap: 54 recusados, 2 aprovados
                │   │               └─ Não: NÓ 1073 | debt_to_income_ratio = '46'? | bootstrap: 277 recusados, 139 aprovados
                │   │                   ├─ Sim: FOLHA 1103 | previsão: APROVADO (1) | bootstrap: 6 recusados, 21 aprovados
                │   │                   └─ Não: NÓ 1074 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 271 recusados, 118 aprovados
                │   │                       ├─ Sim: FOLHA 1102 | previsão: RECUSADO (0) | bootstrap: 41 recusados, 13 aprovados
                │   │                       └─ Não: NÓ 1075 | debt_to_income_ratio = '37'? | bootstrap: 230 recusados, 105 aprovados
                │   │                           ├─ Sim: FOLHA 1101 | previsão: APROVADO (1) | bootstrap: 8 recusados, 10 aprovados
                │   │                           └─ Não: NÓ 1076 | applicant_age = '>74'? | bootstrap: 222 recusados, 95 aprovados
                │   │                               ├─ Sim: FOLHA 1100 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 1 aprovados
                │   │                               └─ Não: NÓ 1077 | debt_to_income_ratio = '40'? | bootstrap: 215 recusados, 94 aprovados
                │   │                                   ├─ Sim: NÓ 1097 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 89.0? | bootstrap: 9 recusados, 8 aprovados
                │   │                                   │   ├─ Sim: FOLHA 1098 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 2 aprovados
                │   │                                   │   └─ Não: FOLHA 1099 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
                │   │                                   └─ Não: NÓ 1078 | applicant_age = '25-34'? | bootstrap: 206 recusados, 86 aprovados
                │   │                                       ├─ Sim: NÓ 1090 | loan_amount (valor solicitado em USD) <= 190000.0? | bootstrap: 72 recusados, 30 aprovados
                │   │                                       │   ├─ Sim: NÓ 1091 | debt_to_income_ratio = '>60%'? | bootstrap: 52 recusados, 11 aprovados
                │   │                                       │   │   ├─ Sim: FOLHA 1095 | previsão: RECUSADO (0) | bootstrap: 23 recusados, 0 aprovados
                │   │                                       │   │   └─ Não: NÓ 1092 | income (renda anual declarada em milhares de USD) <= 91.5? | bootstrap: 29 recusados, 11 aprovados
                │   │                                       │   │       ├─ Sim: FOLHA 1093 | previsão: RECUSADO (0) | bootstrap: 15 recusados, 9 aprovados
                │   │                                       │   │       └─ Não: FOLHA 1094 | previsão: RECUSADO (0) | bootstrap: 14 recusados, 2 aprovados
                │   │                                       │   └─ Não: FOLHA 1096 | previsão: RECUSADO (0) | bootstrap: 20 recusados, 19 aprovados
                │   │                                       └─ Não: NÓ 1079 | applicant_age = '55-64'? | bootstrap: 134 recusados, 56 aprovados
                │   │                                           ├─ Sim: FOLHA 1089 | previsão: RECUSADO (0) | bootstrap: 31 recusados, 12 aprovados
                │   │                                           └─ Não: NÓ 1080 | debt_to_income_ratio = '39'? | bootstrap: 103 recusados, 44 aprovados
                │   │                                               ├─ Sim: FOLHA 1088 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
                │   │                                               └─ Não: NÓ 1081 | applicant_age = '65-74'? | bootstrap: 100 recusados, 39 aprovados
                │   │                                                   ├─ Sim: NÓ 1085 | loan_amount (valor solicitado em USD) <= 170000.0? | bootstrap: 12 recusados, 4 aprovados
                │   │                                                   │   ├─ Sim: FOLHA 1086 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 0 aprovados
                │   │                                                   │   └─ Não: FOLHA 1087 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 4 aprovados
                │   │                                                   └─ Não: NÓ 1082 | property_value (valor do imóvel em USD) <= 170000.0? | bootstrap: 88 recusados, 35 aprovados
                │   │                                                       ├─ Sim: FOLHA 1083 | previsão: RECUSADO (0) | bootstrap: 56 recusados, 4 aprovados
                │   │                                                       └─ Não: FOLHA 1084 | previsão: RECUSADO (0) | bootstrap: 32 recusados, 31 aprovados
                │   └─ Não: NÓ 449 | construction_method = '1'? | bootstrap: 98471 recusados, 791005 aprovados
                │       ├─ Sim: NÓ 515 | lei = '549300KBWX4NV5Q1E376'? | bootstrap: 64498 recusados, 755863 aprovados
                │       │   ├─ Sim: NÓ 847 | debt_to_income_ratio = '47'? | bootstrap: 654 recusados, 3428 aprovados
                │       │   │   ├─ Sim: NÓ 947 | applicant_age = '25-34'? | bootstrap: 8 recusados, 236 aprovados
                │       │   │   │   ├─ Sim: NÓ 955 | income (renda anual declarada em milhares de USD) <= 102.5? | bootstrap: 2 recusados, 85 aprovados
                │       │   │   │   │   ├─ Sim: FOLHA 956 | previsão: APROVADO (1) | bootstrap: 0 recusados, 50 aprovados
                │       │   │   │   │   └─ Não: NÓ 957 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 114.5? | bootstrap: 2 recusados, 35 aprovados
                │       │   │   │   │       ├─ Sim: FOLHA 958 | previsão: APROVADO (1) | bootstrap: 2 recusados, 22 aprovados
                │       │   │   │   │       └─ Não: FOLHA 959 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
                │       │   │   │   └─ Não: NÓ 948 | income (renda anual declarada em milhares de USD) <= 59.0? | bootstrap: 6 recusados, 151 aprovados
                │       │   │   │       ├─ Sim: NÓ 949 | loan_amount (valor solicitado em USD) <= 225000.0? | bootstrap: 2 recusados, 18 aprovados
                │       │   │   │       │   ├─ Sim: FOLHA 950 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │       │   │   │       │   └─ Não: FOLHA 951 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                │       │   │   │       └─ Não: NÓ 952 | applicant_age = '55-64'? | bootstrap: 4 recusados, 133 aprovados
                │       │   │   │           ├─ Sim: FOLHA 954 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
                │       │   │   │           └─ Não: FOLHA 953 | previsão: APROVADO (1) | bootstrap: 4 recusados, 108 aprovados
                │       │   │   └─ Não: NÓ 848 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 646 recusados, 3192 aprovados
                │       │   │       ├─ Sim: NÓ 942 | loan_type = '1'? | bootstrap: 10 recusados, 551 aprovados
                │       │   │       │   ├─ Sim: NÓ 944 | property_value (valor do imóvel em USD) <= 250000.0? | bootstrap: 1 recusados, 425 aprovados
                │       │   │       │   │   ├─ Sim: FOLHA 945 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                │       │   │       │   │   └─ Não: FOLHA 946 | previsão: APROVADO (1) | bootstrap: 0 recusados, 419 aprovados
                │       │   │       │   └─ Não: FOLHA 943 | previsão: APROVADO (1) | bootstrap: 9 recusados, 126 aprovados
                │       │   │       └─ Não: NÓ 849 | debt_to_income_ratio = '37'? | bootstrap: 636 recusados, 2641 aprovados
                │       │   │           ├─ Sim: NÓ 939 | applicant_age = '65-74'? | bootstrap: 10 recusados, 111 aprovados
                │       │   │           │   ├─ Sim: FOLHA 941 | previsão: APROVADO (1) | bootstrap: 2 recusados, 11 aprovados
                │       │   │           │   └─ Não: FOLHA 940 | previsão: APROVADO (1) | bootstrap: 8 recusados, 100 aprovados
                │       │   │           └─ Não: NÓ 850 | debt_to_income_ratio = '49'? | bootstrap: 626 recusados, 2530 aprovados
                │       │   │               ├─ Sim: NÓ 894 | derived_ethnicity = 'Joint'? | bootstrap: 52 recusados, 474 aprovados
                │       │   │               │   ├─ Sim: FOLHA 938 | previsão: APROVADO (1) | bootstrap: 1 recusados, 11 aprovados
                │       │   │               │   └─ Não: NÓ 895 | applicant_age = '65-74'? | bootstrap: 51 recusados, 463 aprovados
                │       │   │               │       ├─ Sim: NÓ 935 | loan_type = '3'? | bootstrap: 1 recusados, 44 aprovados
                │       │   │               │       │   ├─ Sim: FOLHA 937 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │       │   │               │       │   └─ Não: FOLHA 936 | previsão: APROVADO (1) | bootstrap: 1 recusados, 36 aprovados
                │       │   │               │       └─ Não: NÓ 896 | loan_type = '1'? | bootstrap: 50 recusados, 419 aprovados
                │       │   │               │           ├─ Sim: NÓ 918 | loan_amount (valor solicitado em USD) <= 515000.0? | bootstrap: 17 recusados, 188 aprovados
                │       │   │               │           │   ├─ Sim: NÓ 919 | loan_term (prazo em meses) <= 300.0? | bootstrap: 17 recusados, 168 aprovados
                │       │   │               │           │   │   ├─ Sim: FOLHA 920 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                │       │   │               │           │   │   └─ Não: NÓ 921 | property_value (valor do imóvel em USD) <= 440000.0? | bootstrap: 17 recusados, 161 aprovados
                │       │   │               │           │   │       ├─ Sim: NÓ 922 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 7 recusados, 128 aprovados
                │       │   │               │           │   │       │   ├─ Sim: NÓ 926 | applicant_age = '25-34'? | bootstrap: 2 recusados, 26 aprovados
                │       │   │               │           │   │       │   │   ├─ Sim: FOLHA 928 | previsão: APROVADO (1) | bootstrap: 1 recusados, 12 aprovados
                │       │   │               │           │   │       │   │   └─ Não: FOLHA 927 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
                │       │   │               │           │   │       │   └─ Não: NÓ 923 | property_value (valor do imóvel em USD) <= 400000.0? | bootstrap: 5 recusados, 102 aprovados
                │       │   │               │           │   │       │       ├─ Sim: FOLHA 924 | previsão: APROVADO (1) | bootstrap: 2 recusados, 73 aprovados
                │       │   │               │           │   │       │       └─ Não: FOLHA 925 | previsão: APROVADO (1) | bootstrap: 3 recusados, 29 aprovados
                │       │   │               │           │   │       └─ Não: NÓ 929 | applicant_age = '55-64'? | bootstrap: 10 recusados, 33 aprovados
                │       │   │               │           │   │           ├─ Sim: FOLHA 933 | previsão: APROVADO (1) | bootstrap: 2 recusados, 11 aprovados
                │       │   │               │           │   │           └─ Não: NÓ 930 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 8 recusados, 22 aprovados
                │       │   │               │           │   │               ├─ Sim: FOLHA 932 | previsão: APROVADO (1) | bootstrap: 5 recusados, 18 aprovados
                │       │   │               │           │   │               └─ Não: FOLHA 931 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │       │   │               │           │   └─ Não: FOLHA 934 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
                │       │   │               │           └─ Não: NÓ 897 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 67.0? | bootstrap: 33 recusados, 231 aprovados
                │       │   │               │               ├─ Sim: NÓ 898 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 8 recusados, 21 aprovados
                │       │   │               │               │   ├─ Sim: NÓ 900 | income (renda anual declarada em milhares de USD) <= 74.0? | bootstrap: 5 recusados, 12 aprovados
                │       │   │               │               │   │   ├─ Sim: FOLHA 901 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │       │   │               │               │   │   └─ Não: FOLHA 902 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
                │       │   │               │               │   └─ Não: FOLHA 899 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
                │       │   │               │               └─ Não: NÓ 903 | income (renda anual declarada em milhares de USD) <= 112.5? | bootstrap: 25 recusados, 210 aprovados
                │       │   │               │                   ├─ Sim: NÓ 904 | applicant_age = '35-44'? | bootstrap: 22 recusados, 143 aprovados
                │       │   │               │                   │   ├─ Sim: NÓ 912 | property_value (valor do imóvel em USD) <= 405000.0? | bootstrap: 3 recusados, 42 aprovados
                │       │   │               │                   │   │   ├─ Sim: NÓ 913 | property_value (valor do imóvel em USD) <= 310000.0? | bootstrap: 1 recusados, 32 aprovados
                │       │   │               │                   │   │   │   ├─ Sim: FOLHA 914 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │       │   │               │                   │   │   │   └─ Não: FOLHA 915 | previsão: APROVADO (1) | bootstrap: 1 recusados, 18 aprovados
                │       │   │               │                   │   │   └─ Não: FOLHA 916 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
                │       │   │               │                   │   └─ Não: NÓ 905 | applicant_age = '25-34'? | bootstrap: 19 recusados, 101 aprovados
                │       │   │               │                   │       ├─ Sim: NÓ 909 | income (renda anual declarada em milhares de USD) <= 94.5? | bootstrap: 12 recusados, 76 aprovados
                │       │   │               │                   │       │   ├─ Sim: FOLHA 910 | previsão: APROVADO (1) | bootstrap: 7 recusados, 61 aprovados
                │       │   │               │                   │       │   └─ Não: FOLHA 911 | previsão: APROVADO (1) | bootstrap: 5 recusados, 15 aprovados
                │       │   │               │                   │       └─ Não: NÓ 906 | property_value (valor do imóvel em USD) <= 300000.0? | bootstrap: 7 recusados, 25 aprovados
                │       │   │               │                   │           ├─ Sim: FOLHA 907 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                │       │   │               │                   │           └─ Não: FOLHA 908 | previsão: APROVADO (1) | bootstrap: 6 recusados, 17 aprovados
                │       │   │               │                   └─ Não: FOLHA 917 | previsão: APROVADO (1) | bootstrap: 3 recusados, 67 aprovados
                │       │   │               └─ Não: NÓ 851 | debt_to_income_ratio = '36'? | bootstrap: 574 recusados, 2056 aprovados
                │       │   │                   ├─ Sim: FOLHA 893 | previsão: APROVADO (1) | bootstrap: 0 recusados, 127 aprovados
                │       │   │                   └─ Não: NÓ 852 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 84.5? | bootstrap: 574 recusados, 1929 aprovados
                │       │   │                       ├─ Sim: FOLHA 853 | previsão: APROVADO (1) | bootstrap: 166 recusados, 405 aprovados
                │       │   │                       └─ Não: NÓ 854 | loan_term (prazo em meses) <= 330.0? | bootstrap: 408 recusados, 1524 aprovados
                │       │   │                           ├─ Sim: FOLHA 855 | previsão: APROVADO (1) | bootstrap: 0 recusados, 61 aprovados
                │       │   │                           └─ Não: NÓ 856 | loan_type = '2'? | bootstrap: 408 recusados, 1463 aprovados
                │       │   │                               ├─ Sim: NÓ 876 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 209 recusados, 423 aprovados
                │       │   │                               │   ├─ Sim: NÓ 884 | debt_to_income_ratio = '38'? | bootstrap: 128 recusados, 318 aprovados
                │       │   │                               │   │   ├─ Sim: NÓ 890 | applicant_age = '25-34'? | bootstrap: 1 recusados, 11 aprovados
                │       │   │                               │   │   │   ├─ Sim: FOLHA 892 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                │       │   │                               │   │   │   └─ Não: FOLHA 891 | previsão: APROVADO (1) | bootstrap: 1 recusados, 4 aprovados
                │       │   │                               │   │   └─ Não: NÓ 885 | applicant_age = '65-74'? | bootstrap: 127 recusados, 307 aprovados
                │       │   │                               │   │       ├─ Sim: FOLHA 889 | previsão: APROVADO (1) | bootstrap: 5 recusados, 17 aprovados
                │       │   │                               │   │       └─ Não: NÓ 886 | applicant_age = '25-34'? | bootstrap: 122 recusados, 290 aprovados
                │       │   │                               │   │           ├─ Sim: FOLHA 888 | previsão: APROVADO (1) | bootstrap: 68 recusados, 147 aprovados
                │       │   │                               │   │           └─ Não: FOLHA 887 | previsão: APROVADO (1) | bootstrap: 54 recusados, 143 aprovados
                │       │   │                               │   └─ Não: NÓ 877 | debt_to_income_ratio = '41'? | bootstrap: 81 recusados, 105 aprovados
                │       │   │                               │       ├─ Sim: FOLHA 883 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
                │       │   │                               │       └─ Não: NÓ 878 | applicant_age = '25-34'? | bootstrap: 78 recusados, 99 aprovados
                │       │   │                               │           ├─ Sim: FOLHA 882 | previsão: APROVADO (1) | bootstrap: 19 recusados, 56 aprovados
                │       │   │                               │           └─ Não: NÓ 879 | debt_to_income_ratio = '46'? | bootstrap: 59 recusados, 43 aprovados
                │       │   │                               │               ├─ Sim: FOLHA 881 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │       │   │                               │               └─ Não: FOLHA 880 | previsão: RECUSADO (0) | bootstrap: 59 recusados, 35 aprovados
                │       │   │                               └─ Não: NÓ 857 | applicant_age = '35-44'? | bootstrap: 199 recusados, 1040 aprovados
                │       │   │                                   ├─ Sim: NÓ 869 | debt_to_income_ratio = '39'? | bootstrap: 55 recusados, 195 aprovados
                │       │   │                                   │   ├─ Sim: FOLHA 875 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │       │   │                                   │   └─ Não: NÓ 870 | loan_amount (valor solicitado em USD) <= 215000.0? | bootstrap: 55 recusados, 181 aprovados
                │       │   │                                   │       ├─ Sim: FOLHA 871 | previsão: APROVADO (1) | bootstrap: 10 recusados, 13 aprovados
                │       │   │                                   │       └─ Não: NÓ 872 | property_value (valor do imóvel em USD) <= 915000.0? | bootstrap: 45 recusados, 168 aprovados
                │       │   │                                   │           ├─ Sim: FOLHA 873 | previsão: APROVADO (1) | bootstrap: 42 recusados, 164 aprovados
                │       │   │                                   │           └─ Não: FOLHA 874 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │       │   │                                   └─ Não: NÓ 858 | debt_to_income_ratio = '44'? | bootstrap: 144 recusados, 845 aprovados
                │       │   │                                       ├─ Sim: NÓ 864 | income (renda anual declarada em milhares de USD) <= 110.5? | bootstrap: 6 recusados, 83 aprovados
                │       │   │                                       │   ├─ Sim: NÓ 865 | property_value (valor do imóvel em USD) <= 270000.0? | bootstrap: 6 recusados, 49 aprovados
                │       │   │                                       │   │   ├─ Sim: FOLHA 866 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
                │       │   │                                       │   │   └─ Não: FOLHA 867 | previsão: APROVADO (1) | bootstrap: 3 recusados, 42 aprovados
                │       │   │                                       │   └─ Não: FOLHA 868 | previsão: APROVADO (1) | bootstrap: 0 recusados, 34 aprovados
                │       │   │                                       └─ Não: NÓ 859 | loan_type = '4'? | bootstrap: 138 recusados, 762 aprovados
                │       │   │                                           ├─ Sim: FOLHA 863 | previsão: APROVADO (1) | bootstrap: 7 recusados, 8 aprovados
                │       │   │                                           └─ Não: NÓ 860 | applicant_age = '<25'? | bootstrap: 131 recusados, 754 aprovados
                │       │   │                                               ├─ Sim: FOLHA 862 | previsão: APROVADO (1) | bootstrap: 11 recusados, 32 aprovados
                │       │   │                                               └─ Não: FOLHA 861 | previsão: APROVADO (1) | bootstrap: 120 recusados, 722 aprovados
                │       │   └─ Não: NÓ 516 | debt_to_income_ratio = '39'? | bootstrap: 63844 recusados, 752435 aprovados
                │       │       ├─ Sim: NÓ 776 | property_value (valor do imóvel em USD) <= 150000.0? | bootstrap: 1531 recusados, 35144 aprovados
                │       │       │   ├─ Sim: NÓ 777 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 221 recusados, 2149 aprovados
                │       │       │   │   ├─ Sim: NÓ 779 | income (renda anual declarada em milhares de USD) <= 31.5? | bootstrap: 169 recusados, 1822 aprovados
                │       │       │   │   │   ├─ Sim: FOLHA 780 | previsão: APROVADO (1) | bootstrap: 36 recusados, 191 aprovados
                │       │       │   │   │   └─ Não: NÓ 781 | income (renda anual declarada em milhares de USD) <= 49.5? | bootstrap: 133 recusados, 1631 aprovados
                │       │       │   │   │       ├─ Sim: NÓ 782 | lei = '549300AQ3T62GXDU7D76'? | bootstrap: 49 recusados, 808 aprovados
                │       │       │   │   │       │   ├─ Sim: FOLHA 796 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
                │       │       │   │   │       │   └─ Não: NÓ 783 | lei = '549300VZVN841I2ILS84'? | bootstrap: 49 recusados, 790 aprovados
                │       │       │   │   │       │       ├─ Sim: FOLHA 795 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
                │       │       │   │   │       │       └─ Não: NÓ 784 | loan_amount (valor solicitado em USD) <= 60000.0? | bootstrap: 49 recusados, 770 aprovados
                │       │       │   │   │       │           ├─ Sim: NÓ 785 | loan_amount (valor solicitado em USD) <= 40000.0? | bootstrap: 12 recusados, 69 aprovados
                │       │       │   │   │       │           │   ├─ Sim: NÓ 786 | applicant_age = '25-34'? | bootstrap: 4 recusados, 44 aprovados
                │       │       │   │   │       │           │   │   ├─ Sim: FOLHA 790 | previsão: APROVADO (1) | bootstrap: 0 recusados, 22 aprovados
                │       │       │   │   │       │           │   │   └─ Não: NÓ 787 | property_value (valor do imóvel em USD) <= 120000.0? | bootstrap: 4 recusados, 22 aprovados
                │       │       │   │   │       │           │   │       ├─ Sim: FOLHA 788 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
                │       │       │   │   │       │           │   │       └─ Não: FOLHA 789 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
                │       │       │   │   │       │           │   └─ Não: NÓ 791 | loan_amount (valor solicitado em USD) <= 50000.0? | bootstrap: 8 recusados, 25 aprovados
                │       │       │   │   │       │           │       ├─ Sim: FOLHA 792 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
                │       │       │   │   │       │           │       └─ Não: FOLHA 793 | previsão: APROVADO (1) | bootstrap: 6 recusados, 16 aprovados
                │       │       │   │   │       │           └─ Não: FOLHA 794 | previsão: APROVADO (1) | bootstrap: 37 recusados, 701 aprovados
                │       │       │   │   │       └─ Não: FOLHA 797 | previsão: APROVADO (1) | bootstrap: 84 recusados, 823 aprovados
                │       │       │   │   └─ Não: FOLHA 778 | previsão: APROVADO (1) | bootstrap: 52 recusados, 327 aprovados
                │       │       │   └─ Não: NÓ 798 | lei = '549300ZLPU75MU7O0T38'? | bootstrap: 1310 recusados, 32995 aprovados
                │       │       │       ├─ Sim: FOLHA 846 | previsão: APROVADO (1) | bootstrap: 5 recusados, 6 aprovados
                │       │       │       └─ Não: NÓ 799 | lei = '549300YIQ7S7Z8PIHE53'? | bootstrap: 1305 recusados, 32989 aprovados
                │       │       │           ├─ Sim: FOLHA 845 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
                │       │       │           └─ Não: NÓ 800 | lei = '5493008JUW0FLRMO0W64'? | bootstrap: 1302 recusados, 32980 aprovados
                │       │       │               ├─ Sim: FOLHA 844 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
                │       │       │               └─ Não: NÓ 801 | lei = '5493005JPZ3LXXMB0S24'? | bootstrap: 1299 recusados, 32973 aprovados
                │       │       │                   ├─ Sim: FOLHA 843 | previsão: APROVADO (1) | bootstrap: 0 recusados, 84 aprovados
                │       │       │                   └─ Não: NÓ 802 | lei = '549300R9S3MVDV4MGF56'? | bootstrap: 1299 recusados, 32889 aprovados
                │       │       │                       ├─ Sim: NÓ 840 | loan_type = '3'? | bootstrap: 6 recusados, 23 aprovados
                │       │       │                       │   ├─ Sim: FOLHA 842 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
                │       │       │                       │   └─ Não: FOLHA 841 | previsão: APROVADO (1) | bootstrap: 3 recusados, 15 aprovados
                │       │       │                       └─ Não: NÓ 803 | lei = '5493001SXWZ4OFP8Z903'? | bootstrap: 1293 recusados, 32866 aprovados
                │       │       │                           ├─ Sim: NÓ 829 | derived_ethnicity = 'Joint'? | bootstrap: 4 recusados, 600 aprovados
                │       │       │                           │   ├─ Sim: FOLHA 839 | previsão: APROVADO (1) | bootstrap: 0 recusados, 44 aprovados
                │       │       │                           │   └─ Não: NÓ 830 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 4 recusados, 556 aprovados
                │       │       │                           │       ├─ Sim: NÓ 834 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 168.0? | bootstrap: 3 recusados, 384 aprovados
                │       │       │                           │       │   ├─ Sim: NÓ 835 | applicant_age = '65-74'? | bootstrap: 2 recusados, 378 aprovados
                │       │       │                           │       │   │   ├─ Sim: FOLHA 837 | previsão: APROVADO (1) | bootstrap: 0 recusados, 30 aprovados
                │       │       │                           │       │   │   └─ Não: FOLHA 836 | previsão: APROVADO (1) | bootstrap: 2 recusados, 348 aprovados
                │       │       │                           │       │   └─ Não: FOLHA 838 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                │       │       │                           │       └─ Não: NÓ 831 | income (renda anual declarada em milhares de USD) <= 102.5? | bootstrap: 1 recusados, 172 aprovados
                │       │       │                           │           ├─ Sim: FOLHA 832 | previsão: APROVADO (1) | bootstrap: 0 recusados, 101 aprovados
                │       │       │                           │           └─ Não: FOLHA 833 | previsão: APROVADO (1) | bootstrap: 1 recusados, 71 aprovados
                │       │       │                           └─ Não: NÓ 804 | lei = '549300DD4R4SYK5RAQ92'? | bootstrap: 1289 recusados, 32266 aprovados
                │       │       │                               ├─ Sim: NÓ 818 | applicant_age = '65-74'? | bootstrap: 8 recusados, 655 aprovados
                │       │       │                               │   ├─ Sim: NÓ 826 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 79.5? | bootstrap: 1 recusados, 39 aprovados
                │       │       │                               │   │   ├─ Sim: FOLHA 827 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
                │       │       │                               │   │   └─ Não: FOLHA 828 | previsão: APROVADO (1) | bootstrap: 0 recusados, 30 aprovados
                │       │       │                               │   └─ Não: NÓ 819 | income (renda anual declarada em milhares de USD) <= 95.5? | bootstrap: 7 recusados, 616 aprovados
                │       │       │                               │       ├─ Sim: NÓ 820 | applicant_age = '>74'? | bootstrap: 1 recusados, 269 aprovados
                │       │       │                               │       │   ├─ Sim: FOLHA 822 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │       │       │                               │       │   └─ Não: FOLHA 821 | previsão: APROVADO (1) | bootstrap: 1 recusados, 261 aprovados
                │       │       │                               │       └─ Não: NÓ 823 | lien_status = '1'? | bootstrap: 6 recusados, 347 aprovados
                │       │       │                               │           ├─ Sim: FOLHA 825 | previsão: APROVADO (1) | bootstrap: 6 recusados, 315 aprovados
                │       │       │                               │           └─ Não: FOLHA 824 | previsão: APROVADO (1) | bootstrap: 0 recusados, 32 aprovados
                │       │       │                               └─ Não: NÓ 805 | lei = '549300AQ3T62GXDU7D76'? | bootstrap: 1281 recusados, 31611 aprovados
                │       │       │                                   ├─ Sim: NÓ 811 | applicant_age = '55-64'? | bootstrap: 7 recusados, 820 aprovados
                │       │       │                                   │   ├─ Sim: NÓ 815 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 108.5? | bootstrap: 1 recusados, 78 aprovados
                │       │       │                                   │   │   ├─ Sim: FOLHA 816 | previsão: APROVADO (1) | bootstrap: 0 recusados, 39 aprovados
                │       │       │                                   │   │   └─ Não: FOLHA 817 | previsão: APROVADO (1) | bootstrap: 1 recusados, 39 aprovados
                │       │       │                                   │   └─ Não: NÓ 812 | applicant_age = '<25'? | bootstrap: 6 recusados, 742 aprovados
                │       │       │                                   │       ├─ Sim: FOLHA 814 | previsão: APROVADO (1) | bootstrap: 3 recusados, 73 aprovados
                │       │       │                                   │       └─ Não: FOLHA 813 | previsão: APROVADO (1) | bootstrap: 3 recusados, 669 aprovados
                │       │       │                                   └─ Não: NÓ 806 | lei = '549300PJ2WT35S77Q208'? | bootstrap: 1274 recusados, 30791 aprovados
                │       │       │                                       ├─ Sim: FOLHA 810 | previsão: APROVADO (1) | bootstrap: 3 recusados, 14 aprovados
                │       │       │                                       └─ Não: NÓ 807 | lei = '5493000YNV8IX4VD3X12'? | bootstrap: 1271 recusados, 30777 aprovados
                │       │       │                                           ├─ Sim: FOLHA 809 | previsão: APROVADO (1) | bootstrap: 6 recusados, 62 aprovados
                │       │       │                                           └─ Não: FOLHA 808 | previsão: APROVADO (1) | bootstrap: 1265 recusados, 30715 aprovados
                │       │       └─ Não: NÓ 517 | lei = '3Y4U8VZURTYWI1W2K376'? | bootstrap: 62313 recusados, 717291 aprovados
                │       │           ├─ Sim: NÓ 733 | debt_to_income_ratio = '36'? | bootstrap: 325 recusados, 1003 aprovados
                │       │           │   ├─ Sim: FOLHA 775 | previsão: APROVADO (1) | bootstrap: 5 recusados, 42 aprovados
                │       │           │   └─ Não: NÓ 734 | income (renda anual declarada em milhares de USD) <= 20.5? | bootstrap: 320 recusados, 961 aprovados
                │       │           │       ├─ Sim: NÓ 735 | loan_amount (valor solicitado em USD) <= 280000.0? | bootstrap: 16 recusados, 2 aprovados
                │       │           │       │   ├─ Sim: FOLHA 736 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 0 aprovados
                │       │           │       │   └─ Não: FOLHA 737 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 2 aprovados
                │       │           │       └─ Não: NÓ 738 | debt_to_income_ratio = '42'? | bootstrap: 304 recusados, 959 aprovados
                │       │           │           ├─ Sim: NÓ 770 | applicant_age = '35-44'? | bootstrap: 11 recusados, 66 aprovados
                │       │           │           │   ├─ Sim: NÓ 772 | property_value (valor do imóvel em USD) <= 1115000.0? | bootstrap: 2 recusados, 23 aprovados
                │       │           │           │   │   ├─ Sim: FOLHA 773 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
                │       │           │           │   │   └─ Não: FOLHA 774 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
                │       │           │           │   └─ Não: FOLHA 771 | previsão: APROVADO (1) | bootstrap: 9 recusados, 43 aprovados
                │       │           │           └─ Não: NÓ 739 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 293 recusados, 893 aprovados
                │       │           │               ├─ Sim: FOLHA 769 | previsão: APROVADO (1) | bootstrap: 36 recusados, 269 aprovados
                │       │           │               └─ Não: NÓ 740 | applicant_age = '65-74'? | bootstrap: 257 recusados, 624 aprovados
                │       │           │                   ├─ Sim: NÓ 762 | debt_to_income_ratio = '>60%'? | bootstrap: 24 recusados, 72 aprovados
                │       │           │                   │   ├─ Sim: FOLHA 768 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 5 aprovados
                │       │           │                   │   └─ Não: NÓ 763 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 147.5? | bootstrap: 5 recusados, 67 aprovados
                │       │           │                   │       ├─ Sim: FOLHA 764 | previsão: APROVADO (1) | bootstrap: 2 recusados, 53 aprovados
                │       │           │                   │       └─ Não: NÓ 765 | income (renda anual declarada em milhares de USD) <= 126.5? | bootstrap: 3 recusados, 14 aprovados
                │       │           │                   │           ├─ Sim: FOLHA 766 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │       │           │                   │           └─ Não: FOLHA 767 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                │       │           │                   └─ Não: NÓ 741 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 233 recusados, 552 aprovados
                │       │           │                       ├─ Sim: NÓ 751 | debt_to_income_ratio = '47'? | bootstrap: 55 recusados, 95 aprovados
                │       │           │                       │   ├─ Sim: FOLHA 761 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 4 aprovados
                │       │           │                       │   └─ Não: NÓ 752 | debt_to_income_ratio = '49'? | bootstrap: 51 recusados, 91 aprovados
                │       │           │                       │       ├─ Sim: NÓ 758 | applicant_age = '35-44'? | bootstrap: 3 recusados, 17 aprovados
                │       │           │                       │       │   ├─ Sim: FOLHA 760 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                │       │           │                       │       │   └─ Não: FOLHA 759 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
                │       │           │                       │       └─ Não: NÓ 753 | debt_to_income_ratio = '44'? | bootstrap: 48 recusados, 74 aprovados
                │       │           │                       │           ├─ Sim: FOLHA 757 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │       │           │                       │           └─ Não: NÓ 754 | debt_to_income_ratio = '50%-60%'? | bootstrap: 48 recusados, 64 aprovados
                │       │           │                       │               ├─ Sim: FOLHA 756 | previsão: RECUSADO (0) | bootstrap: 21 recusados, 6 aprovados
                │       │           │                       │               └─ Não: FOLHA 755 | previsão: APROVADO (1) | bootstrap: 27 recusados, 58 aprovados
                │       │           │                       └─ Não: NÓ 742 | lien_status = '1'? | bootstrap: 178 recusados, 457 aprovados
                │       │           │                           ├─ Sim: NÓ 744 | applicant_age = '<25'? | bootstrap: 160 recusados, 440 aprovados
                │       │           │                           │   ├─ Sim: FOLHA 750 | previsão: APROVADO (1) | bootstrap: 7 recusados, 9 aprovados
                │       │           │                           │   └─ Não: NÓ 745 | loan_type = '2'? | bootstrap: 153 recusados, 431 aprovados
                │       │           │                           │       ├─ Sim: FOLHA 749 | previsão: APROVADO (1) | bootstrap: 4 recusados, 9 aprovados
                │       │           │                           │       └─ Não: NÓ 746 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 87.5? | bootstrap: 149 recusados, 422 aprovados
                │       │           │                           │           ├─ Sim: FOLHA 747 | previsão: APROVADO (1) | bootstrap: 5 recusados, 65 aprovados
                │       │           │                           │           └─ Não: FOLHA 748 | previsão: APROVADO (1) | bootstrap: 144 recusados, 357 aprovados
                │       │           │                           └─ Não: FOLHA 743 | previsão: RECUSADO (0) | bootstrap: 18 recusados, 17 aprovados
                │       │           └─ Não: NÓ 518 | lei = '549300SGEJUQK4Q7ZY34'? | bootstrap: 61988 recusados, 716288 aprovados
                │       │               ├─ Sim: NÓ 726 | income (renda anual declarada em milhares de USD) <= 47.5? | bootstrap: 56 recusados, 82 aprovados
                │       │               │   ├─ Sim: FOLHA 727 | previsão: RECUSADO (0) | bootstrap: 15 recusados, 2 aprovados
                │       │               │   └─ Não: NÓ 728 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 90.0? | bootstrap: 41 recusados, 80 aprovados
                │       │               │       ├─ Sim: FOLHA 729 | previsão: APROVADO (1) | bootstrap: 22 recusados, 25 aprovados
                │       │               │       └─ Não: NÓ 730 | property_value (valor do imóvel em USD) <= 210000.0? | bootstrap: 19 recusados, 55 aprovados
                │       │               │           ├─ Sim: FOLHA 731 | previsão: APROVADO (1) | bootstrap: 1 recusados, 23 aprovados
                │       │               │           └─ Não: FOLHA 732 | previsão: APROVADO (1) | bootstrap: 18 recusados, 32 aprovados
                │       │               └─ Não: NÓ 519 | lei = '549300H3IZO24NSOO931'? | bootstrap: 61932 recusados, 716206 aprovados
                │       │                   ├─ Sim: NÓ 629 | derived_ethnicity = 'Joint'? | bootstrap: 1623 recusados, 9872 aprovados
                │       │                   │   ├─ Sim: NÓ 687 | debt_to_income_ratio = '41'? | bootstrap: 93 recusados, 777 aprovados
                │       │                   │   │   ├─ Sim: NÓ 721 | loan_type = '1'? | bootstrap: 1 recusados, 40 aprovados
                │       │                   │   │   │   ├─ Sim: NÓ 723 | income (renda anual declarada em milhares de USD) <= 113.5? | bootstrap: 1 recusados, 25 aprovados
                │       │                   │   │   │   │   ├─ Sim: FOLHA 724 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
                │       │                   │   │   │   │   └─ Não: FOLHA 725 | previsão: APROVADO (1) | bootstrap: 0 recusados, 20 aprovados
                │       │                   │   │   │   └─ Não: FOLHA 722 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                │       │                   │   │   └─ Não: NÓ 688 | debt_to_income_ratio = '44'? | bootstrap: 92 recusados, 737 aprovados
                │       │                   │   │       ├─ Sim: NÓ 712 | loan_type = '1'? | bootstrap: 3 recusados, 54 aprovados
                │       │                   │   │       │   ├─ Sim: NÓ 718 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 92.0? | bootstrap: 2 recusados, 28 aprovados
                │       │                   │   │       │   │   ├─ Sim: FOLHA 719 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
                │       │                   │   │       │   │   └─ Não: FOLHA 720 | previsão: APROVADO (1) | bootstrap: 0 recusados, 23 aprovados
                │       │                   │   │       │   └─ Não: NÓ 713 | applicant_age = '35-44'? | bootstrap: 1 recusados, 26 aprovados
                │       │                   │   │       │       ├─ Sim: FOLHA 717 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                │       │                   │   │       │       └─ Não: NÓ 714 | property_value (valor do imóvel em USD) <= 295000.0? | bootstrap: 1 recusados, 19 aprovados
                │       │                   │   │       │           ├─ Sim: FOLHA 715 | previsão: APROVADO (1) | bootstrap: 1 recusados, 5 aprovados
                │       │                   │   │       │           └─ Não: FOLHA 716 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │       │                   │   │       └─ Não: NÓ 689 | applicant_age = '>74'? | bootstrap: 89 recusados, 683 aprovados
                │       │                   │   │           ├─ Sim: FOLHA 711 | previsão: APROVADO (1) | bootstrap: 0 recusados, 14 aprovados
                │       │                   │   │           └─ Não: NÓ 690 | debt_to_income_ratio = '50%-60%'? | bootstrap: 89 recusados, 669 aprovados
                │       │                   │   │               ├─ Sim: NÓ 704 | applicant_age = '65-74'? | bootstrap: 18 recusados, 195 aprovados
                │       │                   │   │               │   ├─ Sim: FOLHA 710 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                │       │                   │   │               │   └─ Não: NÓ 705 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 195.0? | bootstrap: 18 recusados, 185 aprovados
                │       │                   │   │               │       ├─ Sim: NÓ 706 | loan_type = '2'? | bootstrap: 15 recusados, 180 aprovados
                │       │                   │   │               │       │   ├─ Sim: FOLHA 708 | previsão: APROVADO (1) | bootstrap: 10 recusados, 118 aprovados
                │       │                   │   │               │       │   └─ Não: FOLHA 707 | previsão: APROVADO (1) | bootstrap: 5 recusados, 62 aprovados
                │       │                   │   │               │       └─ Não: FOLHA 709 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
                │       │                   │   │               └─ Não: NÓ 691 | loan_amount (valor solicitado em USD) <= 380000.0? | bootstrap: 71 recusados, 474 aprovados
                │       │                   │   │                   ├─ Sim: NÓ 692 | loan_type = '2'? | bootstrap: 56 recusados, 260 aprovados
                │       │                   │   │                   │   ├─ Sim: NÓ 696 | loan_amount (valor solicitado em USD) <= 360000.0? | bootstrap: 38 recusados, 119 aprovados
                │       │                   │   │                   │   │   ├─ Sim: FOLHA 697 | previsão: APROVADO (1) | bootstrap: 29 recusados, 117 aprovados
                │       │                   │   │                   │   │   └─ Não: FOLHA 698 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 2 aprovados
                │       │                   │   │                   │   └─ Não: NÓ 693 | debt_to_income_ratio = '48'? | bootstrap: 18 recusados, 141 aprovados
                │       │                   │   │                   │       ├─ Sim: FOLHA 695 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
                │       │                   │   │                   │       └─ Não: FOLHA 694 | previsão: APROVADO (1) | bootstrap: 18 recusados, 129 aprovados
                │       │                   │   │                   └─ Não: NÓ 699 | debt_to_income_ratio = '37'? | bootstrap: 15 recusados, 214 aprovados
                │       │                   │   │                       ├─ Sim: FOLHA 703 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                │       │                   │   │                       └─ Não: NÓ 700 | property_value (valor do imóvel em USD) <= 570000.0? | bootstrap: 15 recusados, 205 aprovados
                │       │                   │   │                           ├─ Sim: FOLHA 701 | previsão: APROVADO (1) | bootstrap: 14 recusados, 124 aprovados
                │       │                   │   │                           └─ Não: FOLHA 702 | previsão: APROVADO (1) | bootstrap: 1 recusados, 81 aprovados
                │       │                   │   └─ Não: NÓ 630 | debt_to_income_ratio = '<20%'? | bootstrap: 1530 recusados, 9095 aprovados
                │       │                   │       ├─ Sim: FOLHA 686 | previsão: APROVADO (1) | bootstrap: 23 recusados, 212 aprovados
                │       │                   │       └─ Não: NÓ 631 | debt_to_income_ratio = '48'? | bootstrap: 1507 recusados, 8883 aprovados
                │       │                   │           ├─ Sim: NÓ 665 | applicant_age = '<25'? | bootstrap: 50 recusados, 538 aprovados
                │       │                   │           │   ├─ Sim: NÓ 679 | loan_type = '1'? | bootstrap: 6 recusados, 38 aprovados
                │       │                   │           │   │   ├─ Sim: FOLHA 685 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 3 aprovados
                │       │                   │           │   │   └─ Não: NÓ 680 | income (renda anual declarada em milhares de USD) <= 70.0? | bootstrap: 2 recusados, 35 aprovados
                │       │                   │           │   │       ├─ Sim: FOLHA 681 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                │       │                   │           │   │       └─ Não: NÓ 682 | income (renda anual declarada em milhares de USD) <= 98.0? | bootstrap: 2 recusados, 20 aprovados
                │       │                   │           │   │           ├─ Sim: FOLHA 683 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
                │       │                   │           │   │           └─ Não: FOLHA 684 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
                │       │                   │           │   └─ Não: NÓ 666 | lien_status = '1'? | bootstrap: 44 recusados, 500 aprovados
                │       │                   │           │       ├─ Sim: NÓ 668 | loan_type = '1'? | bootstrap: 43 recusados, 489 aprovados
                │       │                   │           │       │   ├─ Sim: NÓ 676 | property_value (valor do imóvel em USD) <= 260000.0? | bootstrap: 21 recusados, 312 aprovados
                │       │                   │           │       │   │   ├─ Sim: FOLHA 677 | previsão: APROVADO (1) | bootstrap: 6 recusados, 13 aprovados
                │       │                   │           │       │   │   └─ Não: FOLHA 678 | previsão: APROVADO (1) | bootstrap: 15 recusados, 299 aprovados
                │       │                   │           │       │   └─ Não: NÓ 669 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 88.5? | bootstrap: 22 recusados, 177 aprovados
                │       │                   │           │       │       ├─ Sim: NÓ 670 | income (renda anual declarada em milhares de USD) <= 72.5? | bootstrap: 12 recusados, 51 aprovados
                │       │                   │           │       │       │   ├─ Sim: FOLHA 671 | previsão: APROVADO (1) | bootstrap: 11 recusados, 12 aprovados
                │       │                   │           │       │       │   └─ Não: FOLHA 672 | previsão: APROVADO (1) | bootstrap: 1 recusados, 39 aprovados
                │       │                   │           │       │       └─ Não: NÓ 673 | property_value (valor do imóvel em USD) <= 450000.0? | bootstrap: 10 recusados, 126 aprovados
                │       │                   │           │       │           ├─ Sim: FOLHA 674 | previsão: APROVADO (1) | bootstrap: 4 recusados, 109 aprovados
                │       │                   │           │       │           └─ Não: FOLHA 675 | previsão: APROVADO (1) | bootstrap: 6 recusados, 17 aprovados
                │       │                   │           │       └─ Não: FOLHA 667 | previsão: APROVADO (1) | bootstrap: 1 recusados, 11 aprovados
                │       │                   │           └─ Não: NÓ 632 | debt_to_income_ratio = '47'? | bootstrap: 1457 recusados, 8345 aprovados
                │       │                   │               ├─ Sim: NÓ 650 | loan_type = '2'? | bootstrap: 27 recusados, 488 aprovados
                │       │                   │               │   ├─ Sim: NÓ 658 | applicant_age = '25-34'? | bootstrap: 12 recusados, 138 aprovados
                │       │                   │               │   │   ├─ Sim: NÓ 660 | income (renda anual declarada em milhares de USD) <= 79.5? | bootstrap: 5 recusados, 79 aprovados
                │       │                   │               │   │   │   ├─ Sim: NÓ 661 | property_value (valor do imóvel em USD) <= 330000.0? | bootstrap: 5 recusados, 38 aprovados
                │       │                   │               │   │   │   │   ├─ Sim: FOLHA 662 | previsão: APROVADO (1) | bootstrap: 1 recusados, 33 aprovados
                │       │                   │               │   │   │   │   └─ Não: FOLHA 663 | previsão: APROVADO (1) | bootstrap: 4 recusados, 5 aprovados
                │       │                   │               │   │   │   └─ Não: FOLHA 664 | previsão: APROVADO (1) | bootstrap: 0 recusados, 41 aprovados
                │       │                   │               │   │   └─ Não: FOLHA 659 | previsão: APROVADO (1) | bootstrap: 7 recusados, 59 aprovados
                │       │                   │               │   └─ Não: NÓ 651 | loan_type = '3'? | bootstrap: 15 recusados, 350 aprovados
                │       │                   │               │       ├─ Sim: FOLHA 657 | previsão: APROVADO (1) | bootstrap: 4 recusados, 42 aprovados
                │       │                   │               │       └─ Não: NÓ 652 | applicant_age = '25-34'? | bootstrap: 11 recusados, 308 aprovados
                │       │                   │               │           ├─ Sim: FOLHA 656 | previsão: APROVADO (1) | bootstrap: 0 recusados, 105 aprovados
                │       │                   │               │           └─ Não: NÓ 653 | property_value (valor do imóvel em USD) <= 430000.0? | bootstrap: 11 recusados, 203 aprovados
                │       │                   │               │               ├─ Sim: FOLHA 654 | previsão: APROVADO (1) | bootstrap: 7 recusados, 82 aprovados
                │       │                   │               │               └─ Não: FOLHA 655 | previsão: APROVADO (1) | bootstrap: 4 recusados, 121 aprovados
                │       │                   │               └─ Não: NÓ 633 | debt_to_income_ratio = '37'? | bootstrap: 1430 recusados, 7857 aprovados
                │       │                   │                   ├─ Sim: FOLHA 649 | previsão: APROVADO (1) | bootstrap: 12 recusados, 325 aprovados
                │       │                   │                   └─ Não: NÓ 634 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 88.5? | bootstrap: 1418 recusados, 7532 aprovados
                │       │                   │                       ├─ Sim: NÓ 635 | debt_to_income_ratio = '42'? | bootstrap: 469 recusados, 1974 aprovados
                │       │                   │                       │   ├─ Sim: NÓ 639 | income (renda anual declarada em milhares de USD) <= 69.5? | bootstrap: 9 recusados, 115 aprovados
                │       │                   │                       │   │   ├─ Sim: FOLHA 640 | previsão: APROVADO (1) | bootstrap: 6 recusados, 38 aprovados
                │       │                   │                       │   │   └─ Não: FOLHA 641 | previsão: APROVADO (1) | bootstrap: 3 recusados, 77 aprovados
                │       │                   │                       │   └─ Não: NÓ 636 | loan_term (prazo em meses) <= 270.0? | bootstrap: 460 recusados, 1859 aprovados
                │       │                   │                       │       ├─ Sim: FOLHA 637 | previsão: APROVADO (1) | bootstrap: 19 recusados, 24 aprovados
                │       │                   │                       │       └─ Não: FOLHA 638 | previsão: APROVADO (1) | bootstrap: 441 recusados, 1835 aprovados
                │       │                   │                       └─ Não: NÓ 642 | debt_to_income_ratio = '41'? | bootstrap: 949 recusados, 5558 aprovados
                │       │                   │                           ├─ Sim: NÓ 646 | applicant_age = '>74'? | bootstrap: 14 recusados, 299 aprovados
                │       │                   │                           │   ├─ Sim: FOLHA 648 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                │       │                   │                           │   └─ Não: FOLHA 647 | previsão: APROVADO (1) | bootstrap: 14 recusados, 291 aprovados
                │       │                   │                           └─ Não: NÓ 643 | applicant_age = '25-34'? | bootstrap: 935 recusados, 5259 aprovados
                │       │                   │                               ├─ Sim: FOLHA 645 | previsão: APROVADO (1) | bootstrap: 389 recusados, 2179 aprovados
                │       │                   │                               └─ Não: FOLHA 644 | previsão: APROVADO (1) | bootstrap: 546 recusados, 3080 aprovados
                │       │                   └─ Não: NÓ 520 | debt_to_income_ratio = '37'? | bootstrap: 60309 recusados, 706334 aprovados
                │       │                       ├─ Sim: NÓ 612 | lei = 'E57ODZWZ7FF32TWEFA76'? | bootstrap: 1240 recusados, 31269 aprovados
                │       │                       │   ├─ Sim: FOLHA 628 | previsão: APROVADO (1) | bootstrap: 6 recusados, 71 aprovados
                │       │                       │   └─ Não: NÓ 613 | lei = 'ZF85QS7OXKPBG52R7N18'? | bootstrap: 1234 recusados, 31198 aprovados
                │       │                       │       ├─ Sim: FOLHA 627 | previsão: APROVADO (1) | bootstrap: 4 recusados, 43 aprovados
                │       │                       │       └─ Não: NÓ 614 | lei = '549300H8JJI69WD7I525'? | bootstrap: 1230 recusados, 31155 aprovados
                │       │                       │           ├─ Sim: FOLHA 626 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
                │       │                       │           └─ Não: NÓ 615 | lei = '549300FX7K8PTEQUU487'? | bootstrap: 1228 recusados, 31143 aprovados
                │       │                       │               ├─ Sim: FOLHA 625 | previsão: APROVADO (1) | bootstrap: 18 recusados, 39 aprovados
                │       │                       │               └─ Não: NÓ 616 | lei = '549300S8SHBRCI7UU819'? | bootstrap: 1210 recusados, 31104 aprovados
                │       │                       │                   ├─ Sim: FOLHA 624 | previsão: APROVADO (1) | bootstrap: 4 recusados, 8 aprovados
                │       │                       │                   └─ Não: NÓ 617 | loan_type = '4'? | bootstrap: 1206 recusados, 31096 aprovados
                │       │                       │                       ├─ Sim: NÓ 621 | lei = '5493001SXWZ4OFP8Z903'? | bootstrap: 83 recusados, 906 aprovados
                │       │                       │                       │   ├─ Sim: FOLHA 623 | previsão: APROVADO (1) | bootstrap: 8 recusados, 12 aprovados
                │       │                       │                       │   └─ Não: FOLHA 622 | previsão: APROVADO (1) | bootstrap: 75 recusados, 894 aprovados
                │       │                       │                       └─ Não: NÓ 618 | lei = '549300SK2GVCQXPD4S58'? | bootstrap: 1123 recusados, 30190 aprovados
                │       │                       │                           ├─ Sim: FOLHA 620 | previsão: APROVADO (1) | bootstrap: 4 recusados, 20 aprovados
                │       │                       │                           └─ Não: FOLHA 619 | previsão: APROVADO (1) | bootstrap: 1119 recusados, 30170 aprovados
                │       │                       └─ Não: NÓ 521 | lien_status = '1'? | bootstrap: 59069 recusados, 675065 aprovados
                │       │                           ├─ Sim: NÓ 551 | total_units = '2'? | bootstrap: 53761 recusados, 647239 aprovados
                │       │                           │   ├─ Sim: NÓ 587 | lei = '5493008XQDQ2YZPQKX53'? | bootstrap: 1204 recusados, 8034 aprovados
                │       │                           │   │   ├─ Sim: FOLHA 611 | previsão: APROVADO (1) | bootstrap: 4 recusados, 6 aprovados
                │       │                           │   │   └─ Não: NÓ 588 | lei = '549300FGXN1K3HLB1R50'? | bootstrap: 1200 recusados, 8028 aprovados
                │       │                           │   │       ├─ Sim: NÓ 600 | loan_type = '1'? | bootstrap: 101 recusados, 240 aprovados
                │       │                           │   │       │   ├─ Sim: NÓ 606 | debt_to_income_ratio = '>60%'? | bootstrap: 53 recusados, 157 aprovados
                │       │                           │   │       │   │   ├─ Sim: FOLHA 610 | previsão: RECUSADO (0) | bootstrap: 18 recusados, 0 aprovados
                │       │                           │   │       │   │   └─ Não: NÓ 607 | debt_to_income_ratio = '48'? | bootstrap: 35 recusados, 157 aprovados
                │       │                           │   │       │   │       ├─ Sim: FOLHA 609 | previsão: APROVADO (1) | bootstrap: 3 recusados, 14 aprovados
                │       │                           │   │       │   │       └─ Não: FOLHA 608 | previsão: APROVADO (1) | bootstrap: 32 recusados, 143 aprovados
                │       │                           │   │       │   └─ Não: NÓ 601 | debt_to_income_ratio = '42'? | bootstrap: 48 recusados, 83 aprovados
                │       │                           │   │       │       ├─ Sim: FOLHA 605 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                │       │                           │   │       │       └─ Não: NÓ 602 | property_value (valor do imóvel em USD) <= 90000.0? | bootstrap: 47 recusados, 75 aprovados
                │       │                           │   │       │           ├─ Sim: FOLHA 603 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 1 aprovados
                │       │                           │   │       │           └─ Não: FOLHA 604 | previsão: APROVADO (1) | bootstrap: 41 recusados, 74 aprovados
                │       │                           │   │       └─ Não: NÓ 589 | debt_to_income_ratio = '50%-60%'? | bootstrap: 1099 recusados, 7788 aprovados
                │       │                           │   │           ├─ Sim: NÓ 595 | lei = '5493008PQSMR41Y70C36'? | bootstrap: 264 recusados, 1328 aprovados
                │       │                           │   │           │   ├─ Sim: FOLHA 599 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
                │       │                           │   │           │   └─ Não: NÓ 596 | lei = '549300VZVN841I2ILS84'? | bootstrap: 262 recusados, 1321 aprovados
                │       │                           │   │           │       ├─ Sim: FOLHA 598 | previsão: APROVADO (1) | bootstrap: 0 recusados, 59 aprovados
                │       │                           │   │           │       └─ Não: FOLHA 597 | previsão: APROVADO (1) | bootstrap: 262 recusados, 1262 aprovados
                │       │                           │   │           └─ Não: NÓ 590 | lei = '5493008NWHQT1R22C024'? | bootstrap: 835 recusados, 6460 aprovados
                │       │                           │   │               ├─ Sim: FOLHA 594 | previsão: APROVADO (1) | bootstrap: 10 recusados, 15 aprovados
                │       │                           │   │               └─ Não: NÓ 591 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 825 recusados, 6445 aprovados
                │       │                           │   │                   ├─ Sim: FOLHA 593 | previsão: APROVADO (1) | bootstrap: 437 recusados, 4520 aprovados
                │       │                           │   │                   └─ Não: FOLHA 592 | previsão: APROVADO (1) | bootstrap: 388 recusados, 1925 aprovados
                │       │                           │   └─ Não: NÓ 552 | lei = 'DRMSV1Q0EKMEXLAU1P80'? | bootstrap: 52557 recusados, 639205 aprovados
                │       │                           │       ├─ Sim: NÓ 574 | debt_to_income_ratio = '50%-60%'? | bootstrap: 382 recusados, 1919 aprovados
                │       │                           │       │   ├─ Sim: NÓ 584 | loan_term (prazo em meses) <= 330.0? | bootstrap: 59 recusados, 164 aprovados
                │       │                           │       │   │   ├─ Sim: FOLHA 585 | previsão: APROVADO (1) | bootstrap: 12 recusados, 112 aprovados
                │       │                           │       │   │   └─ Não: FOLHA 586 | previsão: APROVADO (1) | bootstrap: 47 recusados, 52 aprovados
                │       │                           │       │   └─ Não: NÓ 575 | debt_to_income_ratio = '46'? | bootstrap: 323 recusados, 1755 aprovados
                │       │                           │       │       ├─ Sim: FOLHA 583 | previsão: APROVADO (1) | bootstrap: 10 recusados, 68 aprovados
                │       │                           │       │       └─ Não: NÓ 576 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 313 recusados, 1687 aprovados
                │       │                           │       │           ├─ Sim: NÓ 580 | loan_term (prazo em meses) <= 330.0? | bootstrap: 43 recusados, 453 aprovados
                │       │                           │       │           │   ├─ Sim: FOLHA 581 | previsão: APROVADO (1) | bootstrap: 15 recusados, 74 aprovados
                │       │                           │       │           │   └─ Não: FOLHA 582 | previsão: APROVADO (1) | bootstrap: 28 recusados, 379 aprovados
                │       │                           │       │           └─ Não: NÓ 577 | loan_amount (valor solicitado em USD) <= 50000.0? | bootstrap: 270 recusados, 1234 aprovados
                │       │                           │       │               ├─ Sim: FOLHA 578 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 5 aprovados
                │       │                           │       │               └─ Não: FOLHA 579 | previsão: APROVADO (1) | bootstrap: 258 recusados, 1229 aprovados
                │       │                           │       └─ Não: NÓ 553 | lei = '254900XN7UWEWK13RO81'? | bootstrap: 52175 recusados, 637286 aprovados
                │       │                           │           ├─ Sim: NÓ 567 | debt_to_income_ratio = '42'? | bootstrap: 24 recusados, 1195 aprovados
                │       │                           │           │   ├─ Sim: FOLHA 573 | previsão: APROVADO (1) | bootstrap: 0 recusados, 92 aprovados
                │       │                           │           │   └─ Não: NÓ 568 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 24 recusados, 1103 aprovados
                │       │                           │           │       ├─ Sim: FOLHA 572 | previsão: APROVADO (1) | bootstrap: 0 recusados, 314 aprovados
                │       │                           │           │       └─ Não: NÓ 569 | income (renda anual declarada em milhares de USD) <= 34.0? | bootstrap: 24 recusados, 789 aprovados
                │       │                           │           │           ├─ Sim: FOLHA 570 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
                │       │                           │           │           └─ Não: FOLHA 571 | previsão: APROVADO (1) | bootstrap: 21 recusados, 783 aprovados
                │       │                           │           └─ Não: NÓ 554 | debt_to_income_ratio = '45'? | bootstrap: 52151 recusados, 636091 aprovados
                │       │                           │               ├─ Sim: NÓ 562 | income (renda anual declarada em milhares de USD) <= 523.5? | bootstrap: 1584 recusados, 33762 aprovados
                │       │                           │               │   ├─ Sim: NÓ 563 | lei = 'C5654JQHZUHN0772B561'? | bootstrap: 1554 recusados, 33649 aprovados
                │       │                           │               │   │   ├─ Sim: FOLHA 565 | previsão: APROVADO (1) | bootstrap: 33 recusados, 105 aprovados
                │       │                           │               │   │   └─ Não: FOLHA 564 | previsão: APROVADO (1) | bootstrap: 1521 recusados, 33544 aprovados
                │       │                           │               │   └─ Não: FOLHA 566 | previsão: APROVADO (1) | bootstrap: 30 recusados, 113 aprovados
                │       │                           │               └─ Não: NÓ 555 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 50567 recusados, 602329 aprovados
                │       │                           │                   ├─ Sim: NÓ 559 | lei = '549300CR5XDK0TP7I727'? | bootstrap: 14448 recusados, 101544 aprovados
                │       │                           │                   │   ├─ Sim: FOLHA 561 | previsão: APROVADO (1) | bootstrap: 44 recusados, 85 aprovados
                │       │                           │                   │   └─ Não: FOLHA 560 | previsão: APROVADO (1) | bootstrap: 14404 recusados, 101459 aprovados
                │       │                           │                   └─ Não: NÓ 556 | debt_to_income_ratio = '44'? | bootstrap: 36119 recusados, 500785 aprovados
                │       │                           │                       ├─ Sim: FOLHA 558 | previsão: APROVADO (1) | bootstrap: 1219 recusados, 31865 aprovados
                │       │                           │                       └─ Não: FOLHA 557 | previsão: APROVADO (1) | bootstrap: 34900 recusados, 468920 aprovados
                │       │                           └─ Não: NÓ 522 | debt_to_income_ratio = '38'? | bootstrap: 5308 recusados, 27826 aprovados
                │       │                               ├─ Sim: FOLHA 550 | previsão: APROVADO (1) | bootstrap: 86 recusados, 1454 aprovados
                │       │                               └─ Não: NÓ 523 | lei = 'AD6GFRVSDT01YPT1CS68'? | bootstrap: 5222 recusados, 26372 aprovados
                │       │                                   ├─ Sim: NÓ 543 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 131 recusados, 179 aprovados
                │       │                                   │   ├─ Sim: NÓ 547 | applicant_age = '25-34'? | bootstrap: 4 recusados, 52 aprovados
                │       │                                   │   │   ├─ Sim: FOLHA 549 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                │       │                                   │   │   └─ Não: FOLHA 548 | previsão: APROVADO (1) | bootstrap: 4 recusados, 43 aprovados
                │       │                                   │   └─ Não: NÓ 544 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 127 recusados, 127 aprovados
                │       │                                   │       ├─ Sim: FOLHA 546 | previsão: RECUSADO (0) | bootstrap: 39 recusados, 11 aprovados
                │       │                                   │       └─ Não: FOLHA 545 | previsão: APROVADO (1) | bootstrap: 88 recusados, 116 aprovados
                │       │                                   └─ Não: NÓ 524 | lei = '549300FX7K8PTEQUU487'? | bootstrap: 5091 recusados, 26193 aprovados
                │       │                                       ├─ Sim: NÓ 536 | debt_to_income_ratio = '36'? | bootstrap: 50 recusados, 50 aprovados
                │       │                                       │   ├─ Sim: FOLHA 542 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                │       │                                       │   └─ Não: NÓ 537 | applicant_age = '65-74'? | bootstrap: 49 recusados, 42 aprovados
                │       │                                       │       ├─ Sim: FOLHA 541 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 5 aprovados
                │       │                                       │       └─ Não: NÓ 538 | debt_to_income_ratio = '<20%'? | bootstrap: 42 recusados, 37 aprovados
                │       │                                       │           ├─ Sim: FOLHA 540 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 2 aprovados
                │       │                                       │           └─ Não: FOLHA 539 | previsão: APROVADO (1) | bootstrap: 30 recusados, 35 aprovados
                │       │                                       └─ Não: NÓ 525 | lei = '5493001XKL0FTNOJ0866'? | bootstrap: 5041 recusados, 26143 aprovados
                │       │                                           ├─ Sim: NÓ 533 | applicant_age = '35-44'? | bootstrap: 2 recusados, 163 aprovados
                │       │                                           │   ├─ Sim: FOLHA 535 | previsão: APROVADO (1) | bootstrap: 0 recusados, 41 aprovados
                │       │                                           │   └─ Não: FOLHA 534 | previsão: APROVADO (1) | bootstrap: 2 recusados, 122 aprovados
                │       │                                           └─ Não: NÓ 526 | applicant_age = '<25'? | bootstrap: 5039 recusados, 25980 aprovados
                │       │                                               ├─ Sim: NÓ 530 | debt_to_income_ratio = '<20%'? | bootstrap: 266 recusados, 3708 aprovados
                │       │                                               │   ├─ Sim: FOLHA 532 | previsão: APROVADO (1) | bootstrap: 42 recusados, 288 aprovados
                │       │                                               │   └─ Não: FOLHA 531 | previsão: APROVADO (1) | bootstrap: 224 recusados, 3420 aprovados
                │       │                                               └─ Não: NÓ 527 | lei = '7DMUJTL9FFTVIAG9H788'? | bootstrap: 4773 recusados, 22272 aprovados
                │       │                                                   ├─ Sim: FOLHA 529 | previsão: RECUSADO (0) | bootstrap: 21 recusados, 3 aprovados
                │       │                                                   └─ Não: FOLHA 528 | previsão: APROVADO (1) | bootstrap: 4752 recusados, 22269 aprovados
                │       └─ Não: NÓ 450 | loan_type = '4'? | bootstrap: 33973 recusados, 35142 aprovados
                │           ├─ Sim: NÓ 510 | applicant_age = '25-34'? | bootstrap: 72 recusados, 217 aprovados
                │           │   ├─ Sim: FOLHA 514 | previsão: APROVADO (1) | bootstrap: 37 recusados, 80 aprovados
                │           │   └─ Não: NÓ 511 | debt_to_income_ratio = '39'? | bootstrap: 35 recusados, 137 aprovados
                │           │       ├─ Sim: FOLHA 513 | previsão: APROVADO (1) | bootstrap: 2 recusados, 20 aprovados
                │           │       └─ Não: FOLHA 512 | previsão: APROVADO (1) | bootstrap: 33 recusados, 117 aprovados
                │           └─ Não: NÓ 451 | applicant_age = '65-74'? | bootstrap: 33901 recusados, 34925 aprovados
                │               ├─ Sim: FOLHA 509 | previsão: APROVADO (1) | bootstrap: 1918 recusados, 3184 aprovados
                │               └─ Não: NÓ 452 | debt_to_income_ratio = '>60%'? | bootstrap: 31983 recusados, 31741 aprovados
                │                   ├─ Sim: FOLHA 508 | previsão: RECUSADO (0) | bootstrap: 9805 recusados, 239 aprovados
                │                   └─ Não: NÓ 453 | lei = '549300BTTR873E71L054'? | bootstrap: 22178 recusados, 31502 aprovados
                │                       ├─ Sim: FOLHA 507 | previsão: APROVADO (1) | bootstrap: 3 recusados, 30 aprovados
                │                       └─ Não: NÓ 454 | debt_to_income_ratio = '50%-60%'? | bootstrap: 22175 recusados, 31472 aprovados
                │                           ├─ Sim: NÓ 484 | lei = '549300MZ8VZJOVC63092'? | bootstrap: 4995 recusados, 5022 aprovados
                │                           │   ├─ Sim: FOLHA 506 | previsão: APROVADO (1) | bootstrap: 12 recusados, 18 aprovados
                │                           │   └─ Não: NÓ 485 | lei = '5493009XSE00M8B9R434'? | bootstrap: 4983 recusados, 5004 aprovados
                │                           │       ├─ Sim: NÓ 495 | loan_type = '1'? | bootstrap: 116 recusados, 12 aprovados
                │                           │       │   ├─ Sim: NÓ 497 | loan_term (prazo em meses) <= 258.0? | bootstrap: 105 recusados, 11 aprovados
                │                           │       │   │   ├─ Sim: NÓ 498 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 100.5? | bootstrap: 50 recusados, 11 aprovados
                │                           │       │   │   │   ├─ Sim: FOLHA 499 | previsão: RECUSADO (0) | bootstrap: 29 recusados, 1 aprovados
                │                           │       │   │   │   └─ Não: NÓ 500 | applicant_age = '35-44'? | bootstrap: 21 recusados, 10 aprovados
                │                           │       │   │   │       ├─ Sim: FOLHA 504 | previsão: APROVADO (1) | bootstrap: 4 recusados, 7 aprovados
                │                           │       │   │   │       └─ Não: NÓ 501 | property_value (valor do imóvel em USD) <= 130000.0? | bootstrap: 17 recusados, 3 aprovados
                │                           │       │   │   │           ├─ Sim: FOLHA 502 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 0 aprovados
                │                           │       │   │   │           └─ Não: FOLHA 503 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 3 aprovados
                │                           │       │   │   └─ Não: FOLHA 505 | previsão: RECUSADO (0) | bootstrap: 55 recusados, 0 aprovados
                │                           │       │   └─ Não: FOLHA 496 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 1 aprovados
                │                           │       └─ Não: NÓ 486 | lei = '5493000YNV8IX4VD3X12'? | bootstrap: 4867 recusados, 4992 aprovados
                │                           │           ├─ Sim: NÓ 488 | loan_term (prazo em meses) <= 330.0? | bootstrap: 928 recusados, 125 aprovados
                │                           │           │   ├─ Sim: FOLHA 489 | previsão: RECUSADO (0) | bootstrap: 882 recusados, 109 aprovados
                │                           │           │   └─ Não: NÓ 490 | loan_amount (valor solicitado em USD) <= 145000.0? | bootstrap: 46 recusados, 16 aprovados
                │                           │           │       ├─ Sim: NÓ 491 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 85.0? | bootstrap: 22 recusados, 1 aprovados
                │                           │           │       │   ├─ Sim: FOLHA 492 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 1 aprovados
                │                           │           │       │   └─ Não: FOLHA 493 | previsão: RECUSADO (0) | bootstrap: 15 recusados, 0 aprovados
                │                           │           │       └─ Não: FOLHA 494 | previsão: RECUSADO (0) | bootstrap: 24 recusados, 15 aprovados
                │                           │           └─ Não: FOLHA 487 | previsão: APROVADO (1) | bootstrap: 3939 recusados, 4867 aprovados
                │                           └─ Não: NÓ 455 | lei = '9845001DAE89414CBB55'? | bootstrap: 17180 recusados, 26450 aprovados
                │                               ├─ Sim: FOLHA 483 | previsão: APROVADO (1) | bootstrap: 0 recusados, 391 aprovados
                │                               └─ Não: NÓ 456 | lei = '549300MP59Z6SRKSVR49'? | bootstrap: 17180 recusados, 26059 aprovados
                │                                   ├─ Sim: FOLHA 482 | previsão: APROVADO (1) | bootstrap: 0 recusados, 54 aprovados
                │                                   └─ Não: NÓ 457 | debt_to_income_ratio = '<20%'? | bootstrap: 17180 recusados, 26005 aprovados
                │                                       ├─ Sim: FOLHA 481 | previsão: RECUSADO (0) | bootstrap: 2354 recusados, 1403 aprovados
                │                                       └─ Não: NÓ 458 | debt_to_income_ratio = '38'? | bootstrap: 14826 recusados, 24602 aprovados
                │                                           ├─ Sim: FOLHA 480 | previsão: APROVADO (1) | bootstrap: 701 recusados, 1472 aprovados
                │                                           └─ Não: NÓ 459 | lei = '549300DD4R4SYK5RAQ92'? | bootstrap: 14125 recusados, 23130 aprovados
                │                                               ├─ Sim: NÓ 471 | applicant_age = '25-34'? | bootstrap: 11 recusados, 335 aprovados
                │                                               │   ├─ Sim: NÓ 477 | loan_amount (valor solicitado em USD) <= 190000.0? | bootstrap: 2 recusados, 139 aprovados
                │                                               │   │   ├─ Sim: FOLHA 478 | previsão: APROVADO (1) | bootstrap: 2 recusados, 62 aprovados
                │                                               │   │   └─ Não: FOLHA 479 | previsão: APROVADO (1) | bootstrap: 0 recusados, 77 aprovados
                │                                               │   └─ Não: NÓ 472 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 9 recusados, 196 aprovados
                │                                               │       ├─ Sim: NÓ 474 | loan_type = '2'? | bootstrap: 6 recusados, 166 aprovados
                │                                               │       │   ├─ Sim: FOLHA 476 | previsão: APROVADO (1) | bootstrap: 1 recusados, 67 aprovados
                │                                               │       │   └─ Não: FOLHA 475 | previsão: APROVADO (1) | bootstrap: 5 recusados, 99 aprovados
                │                                               │       └─ Não: FOLHA 473 | previsão: APROVADO (1) | bootstrap: 3 recusados, 30 aprovados
                │                                               └─ Não: NÓ 460 | lei = '549300GKFNPRWNS0GF29'? | bootstrap: 14114 recusados, 22795 aprovados
                │                                                   ├─ Sim: NÓ 468 | debt_to_income_ratio = '40'? | bootstrap: 14 recusados, 227 aprovados
                │                                                   │   ├─ Sim: FOLHA 470 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
                │                                                   │   └─ Não: FOLHA 469 | previsão: APROVADO (1) | bootstrap: 14 recusados, 208 aprovados
                │                                                   └─ Não: NÓ 461 | applicant_age = '55-64'? | bootstrap: 14100 recusados, 22568 aprovados
                │                                                       ├─ Sim: NÓ 465 | loan_term (prazo em meses) <= 320.5? | bootstrap: 1668 recusados, 3669 aprovados
                │                                                       │   ├─ Sim: FOLHA 466 | previsão: APROVADO (1) | bootstrap: 1362 recusados, 1948 aprovados
                │                                                       │   └─ Não: FOLHA 467 | previsão: APROVADO (1) | bootstrap: 306 recusados, 1721 aprovados
                │                                                       └─ Não: NÓ 462 | lei = '549300RYBJHWWDENV610'? | bootstrap: 12432 recusados, 18899 aprovados
                │                                                           ├─ Sim: FOLHA 464 | previsão: APROVADO (1) | bootstrap: 0 recusados, 34 aprovados
                │                                                           └─ Não: FOLHA 463 | previsão: APROVADO (1) | bootstrap: 12432 recusados, 18865 aprovados
                └─ Não: NÓ 5 | lei = '549300GKFNPRWNS0GF29'? | bootstrap: 44859 recusados, 229292 aprovados
                    ├─ Sim: NÓ 363 | debt_to_income_ratio = '>60%'? | bootstrap: 233 recusados, 3982 aprovados
                    │   ├─ Sim: NÓ 445 | loan_amount (valor solicitado em USD) <= 630000.0? | bootstrap: 63 recusados, 27 aprovados
                    │   │   ├─ Sim: FOLHA 446 | previsão: RECUSADO (0) | bootstrap: 58 recusados, 18 aprovados
                    │   │   └─ Não: FOLHA 447 | previsão: APROVADO (1) | bootstrap: 5 recusados, 9 aprovados
                    │   └─ Não: NÓ 364 | total_units = '3'? | bootstrap: 170 recusados, 3955 aprovados
                    │       ├─ Sim: FOLHA 444 | previsão: APROVADO (1) | bootstrap: 6 recusados, 15 aprovados
                    │       └─ Não: NÓ 365 | applicant_age = '55-64'? | bootstrap: 164 recusados, 3940 aprovados
                    │           ├─ Sim: NÓ 407 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 39 recusados, 425 aprovados
                    │           │   ├─ Sim: NÓ 441 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 83.0? | bootstrap: 1 recusados, 24 aprovados
                    │           │   │   ├─ Sim: FOLHA 442 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                    │           │   │   └─ Não: FOLHA 443 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
                    │           │   └─ Não: NÓ 408 | debt_to_income_ratio = '45'? | bootstrap: 38 recusados, 401 aprovados
                    │           │       ├─ Sim: FOLHA 440 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
                    │           │       └─ Não: NÓ 409 | debt_to_income_ratio = '41'? | bootstrap: 38 recusados, 373 aprovados
                    │           │           ├─ Sim: FOLHA 439 | previsão: APROVADO (1) | bootstrap: 1 recusados, 18 aprovados
                    │           │           └─ Não: NÓ 410 | debt_to_income_ratio = '<20%'? | bootstrap: 37 recusados, 355 aprovados
                    │           │               ├─ Sim: FOLHA 438 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                    │           │               └─ Não: NÓ 411 | loan_type = '2'? | bootstrap: 37 recusados, 346 aprovados
                    │           │                   ├─ Sim: NÓ 423 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 12 recusados, 113 aprovados
                    │           │                   │   ├─ Sim: NÓ 425 | debt_to_income_ratio = '47'? | bootstrap: 8 recusados, 98 aprovados
                    │           │                   │   │   ├─ Sim: FOLHA 437 | previsão: APROVADO (1) | bootstrap: 0 recusados, 16 aprovados
                    │           │                   │   │   └─ Não: NÓ 426 | income (renda anual declarada em milhares de USD) <= 188.0? | bootstrap: 8 recusados, 82 aprovados
                    │           │                   │   │       ├─ Sim: NÓ 427 | loan_amount (valor solicitado em USD) <= 280000.0? | bootstrap: 6 recusados, 76 aprovados
                    │           │                   │   │       │   ├─ Sim: NÓ 428 | debt_to_income_ratio = '50%-60%'? | bootstrap: 4 recusados, 22 aprovados
                    │           │                   │   │       │   │   ├─ Sim: FOLHA 430 | previsão: APROVADO (1) | bootstrap: 3 recusados, 15 aprovados
                    │           │                   │   │       │   │   └─ Não: FOLHA 429 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
                    │           │                   │   │       │   └─ Não: NÓ 431 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 127.0? | bootstrap: 2 recusados, 54 aprovados
                    │           │                   │   │       │       ├─ Sim: NÓ 432 | loan_amount (valor solicitado em USD) <= 360000.0? | bootstrap: 1 recusados, 48 aprovados
                    │           │                   │   │       │       │   ├─ Sim: FOLHA 433 | previsão: APROVADO (1) | bootstrap: 0 recusados, 26 aprovados
                    │           │                   │   │       │       │   └─ Não: FOLHA 434 | previsão: APROVADO (1) | bootstrap: 1 recusados, 22 aprovados
                    │           │                   │   │       │       └─ Não: FOLHA 435 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                    │           │                   │   │       └─ Não: FOLHA 436 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
                    │           │                   │   └─ Não: FOLHA 424 | previsão: APROVADO (1) | bootstrap: 4 recusados, 15 aprovados
                    │           │                   └─ Não: NÓ 412 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 25 recusados, 233 aprovados
                    │           │                       ├─ Sim: FOLHA 422 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                    │           │                       └─ Não: NÓ 413 | debt_to_income_ratio = '49'? | bootstrap: 25 recusados, 226 aprovados
                    │           │                           ├─ Sim: FOLHA 421 | previsão: APROVADO (1) | bootstrap: 3 recusados, 39 aprovados
                    │           │                           └─ Não: NÓ 414 | debt_to_income_ratio = '47'? | bootstrap: 22 recusados, 187 aprovados
                    │           │                               ├─ Sim: FOLHA 420 | previsão: APROVADO (1) | bootstrap: 1 recusados, 14 aprovados
                    │           │                               └─ Não: NÓ 415 | derived_race = 'Joint'? | bootstrap: 21 recusados, 173 aprovados
                    │           │                                   ├─ Sim: FOLHA 419 | previsão: APROVADO (1) | bootstrap: 0 recusados, 36 aprovados
                    │           │                                   └─ Não: NÓ 416 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 48.5? | bootstrap: 21 recusados, 137 aprovados
                    │           │                                       ├─ Sim: FOLHA 417 | previsão: APROVADO (1) | bootstrap: 5 recusados, 9 aprovados
                    │           │                                       └─ Não: FOLHA 418 | previsão: APROVADO (1) | bootstrap: 16 recusados, 128 aprovados
                    │           └─ Não: NÓ 366 | debt_to_income_ratio = '48'? | bootstrap: 125 recusados, 3515 aprovados
                    │               ├─ Sim: NÓ 404 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 6 recusados, 218 aprovados
                    │               │   ├─ Sim: FOLHA 406 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                    │               │   └─ Não: FOLHA 405 | previsão: APROVADO (1) | bootstrap: 6 recusados, 203 aprovados
                    │               └─ Não: NÓ 367 | debt_to_income_ratio = '50%-60%'? | bootstrap: 119 recusados, 3297 aprovados
                    │                   ├─ Sim: NÓ 401 | lien_status = '1'? | bootstrap: 33 recusados, 524 aprovados
                    │                   │   ├─ Sim: FOLHA 403 | previsão: APROVADO (1) | bootstrap: 27 recusados, 483 aprovados
                    │                   │   └─ Não: FOLHA 402 | previsão: APROVADO (1) | bootstrap: 6 recusados, 41 aprovados
                    │                   └─ Não: NÓ 368 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 86 recusados, 2773 aprovados
                    │                       ├─ Sim: FOLHA 400 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                    │                       └─ Não: NÓ 369 | loan_type = '1'? | bootstrap: 86 recusados, 2758 aprovados
                    │                           ├─ Sim: NÓ 395 | debt_to_income_ratio = '47'? | bootstrap: 40 recusados, 1703 aprovados
                    │                           │   ├─ Sim: NÓ 397 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 127.0? | bootstrap: 3 recusados, 143 aprovados
                    │                           │   │   ├─ Sim: FOLHA 398 | previsão: APROVADO (1) | bootstrap: 0 recusados, 91 aprovados
                    │                           │   │   └─ Não: FOLHA 399 | previsão: APROVADO (1) | bootstrap: 3 recusados, 52 aprovados
                    │                           │   └─ Não: FOLHA 396 | previsão: APROVADO (1) | bootstrap: 37 recusados, 1560 aprovados
                    │                           └─ Não: NÓ 370 | loan_type = '3'? | bootstrap: 46 recusados, 1055 aprovados
                    │                               ├─ Sim: NÓ 388 | total_units = '1'? | bootstrap: 2 recusados, 245 aprovados
                    │                               │   ├─ Sim: NÓ 390 | income (renda anual declarada em milhares de USD) <= 195.0? | bootstrap: 2 recusados, 236 aprovados
                    │                               │   │   ├─ Sim: NÓ 391 | debt_to_income_ratio = '49'? | bootstrap: 1 recusados, 198 aprovados
                    │                               │   │   │   ├─ Sim: FOLHA 393 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
                    │                               │   │   │   └─ Não: FOLHA 392 | previsão: APROVADO (1) | bootstrap: 1 recusados, 180 aprovados
                    │                               │   │   └─ Não: FOLHA 394 | previsão: APROVADO (1) | bootstrap: 1 recusados, 38 aprovados
                    │                               │   └─ Não: FOLHA 389 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                    │                               └─ Não: NÓ 371 | applicant_age = '<25'? | bootstrap: 44 recusados, 810 aprovados
                    │                                   ├─ Sim: FOLHA 387 | previsão: APROVADO (1) | bootstrap: 6 recusados, 60 aprovados
                    │                                   └─ Não: NÓ 372 | loan_type = '2'? | bootstrap: 38 recusados, 750 aprovados
                    │                                       ├─ Sim: NÓ 384 | construction_method = '2'? | bootstrap: 21 recusados, 472 aprovados
                    │                                       │   ├─ Sim: FOLHA 386 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
                    │                                       │   └─ Não: FOLHA 385 | previsão: APROVADO (1) | bootstrap: 21 recusados, 459 aprovados
                    │                                       └─ Não: NÓ 373 | derived_race = 'Asian'? | bootstrap: 17 recusados, 278 aprovados
                    │                                           ├─ Sim: FOLHA 383 | previsão: APROVADO (1) | bootstrap: 0 recusados, 55 aprovados
                    │                                           └─ Não: NÓ 374 | derived_ethnicity = 'Joint'? | bootstrap: 17 recusados, 223 aprovados
                    │                                               ├─ Sim: FOLHA 382 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
                    │                                               └─ Não: NÓ 375 | debt_to_income_ratio = '38'? | bootstrap: 17 recusados, 204 aprovados
                    │                                                   ├─ Sim: NÓ 379 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 88.5? | bootstrap: 3 recusados, 14 aprovados
                    │                                                   │   ├─ Sim: FOLHA 380 | previsão: APROVADO (1) | bootstrap: 2 recusados, 5 aprovados
                    │                                                   │   └─ Não: FOLHA 381 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
                    │                                                   └─ Não: NÓ 376 | debt_to_income_ratio = '47'? | bootstrap: 14 recusados, 190 aprovados
                    │                                                       ├─ Sim: FOLHA 378 | previsão: APROVADO (1) | bootstrap: 1 recusados, 31 aprovados
                    │                                                       └─ Não: FOLHA 377 | previsão: APROVADO (1) | bootstrap: 13 recusados, 159 aprovados
                    └─ Não: NÓ 6 | lei = '549300Z4E78BSM0XM758'? | bootstrap: 44626 recusados, 225310 aprovados
                        ├─ Sim: NÓ 344 | applicant_age = '55-64'? | bootstrap: 98 recusados, 58 aprovados
                        │   ├─ Sim: NÓ 354 | loan_term (prazo em meses) <= 90.0? | bootstrap: 38 recusados, 15 aprovados
                        │   │   ├─ Sim: FOLHA 355 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 2 aprovados
                        │   │   └─ Não: NÓ 356 | lien_status = '2'? | bootstrap: 28 recusados, 13 aprovados
                        │   │       ├─ Sim: NÓ 360 | derived_race = 'Joint'? | bootstrap: 17 recusados, 9 aprovados
                        │   │       │   ├─ Sim: FOLHA 362 | previsão: RECUSADO (0) | bootstrap: 7 recusados, 7 aprovados
                        │   │       │   └─ Não: FOLHA 361 | previsão: RECUSADO (0) | bootstrap: 10 recusados, 2 aprovados
                        │   │       └─ Não: NÓ 357 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 62.5? | bootstrap: 11 recusados, 4 aprovados
                        │   │           ├─ Sim: FOLHA 358 | previsão: RECUSADO (0) | bootstrap: 6 recusados, 1 aprovados
                        │   │           └─ Não: FOLHA 359 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 3 aprovados
                        │   └─ Não: NÓ 345 | debt_to_income_ratio = '45'? | bootstrap: 60 recusados, 43 aprovados
                        │       ├─ Sim: FOLHA 353 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 4 aprovados
                        │       └─ Não: NÓ 346 | loan_term (prazo em meses) <= 210.0? | bootstrap: 55 recusados, 39 aprovados
                        │           ├─ Sim: NÓ 347 | loan_term (prazo em meses) <= 150.0? | bootstrap: 12 recusados, 15 aprovados
                        │           │   ├─ Sim: FOLHA 348 | previsão: RECUSADO (0) | bootstrap: 11 recusados, 3 aprovados
                        │           │   └─ Não: FOLHA 349 | previsão: APROVADO (1) | bootstrap: 1 recusados, 12 aprovados
                        │           └─ Não: NÓ 350 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 53.0? | bootstrap: 43 recusados, 24 aprovados
                        │               ├─ Sim: FOLHA 351 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 0 aprovados
                        │               └─ Não: FOLHA 352 | previsão: RECUSADO (0) | bootstrap: 35 recusados, 24 aprovados
                        └─ Não: NÓ 7 | lei = '549300DD5QQUHO6PCH70'? | bootstrap: 44528 recusados, 225252 aprovados
                            ├─ Sim: NÓ 265 | debt_to_income_ratio = '44'? | bootstrap: 273 recusados, 4642 aprovados
                            │   ├─ Sim: NÓ 329 | derived_race = 'Black or African American'? | bootstrap: 14 recusados, 242 aprovados
                            │   │   ├─ Sim: NÓ 333 | loan_type = '2'? | bootstrap: 7 recusados, 146 aprovados
                            │   │   │   ├─ Sim: FOLHA 343 | previsão: APROVADO (1) | bootstrap: 2 recusados, 14 aprovados
                            │   │   │   └─ Não: NÓ 334 | income (renda anual declarada em milhares de USD) <= 68.5? | bootstrap: 5 recusados, 132 aprovados
                            │   │   │       ├─ Sim: NÓ 335 | applicant_age = '<25'? | bootstrap: 5 recusados, 38 aprovados
                            │   │   │       │   ├─ Sim: FOLHA 341 | previsão: APROVADO (1) | bootstrap: 1 recusados, 8 aprovados
                            │   │   │       │   └─ Não: NÓ 336 | applicant_age = '25-34'? | bootstrap: 4 recusados, 30 aprovados
                            │   │   │       │       ├─ Sim: FOLHA 340 | previsão: APROVADO (1) | bootstrap: 1 recusados, 13 aprovados
                            │   │   │       │       └─ Não: NÓ 337 | income (renda anual declarada em milhares de USD) <= 59.5? | bootstrap: 3 recusados, 17 aprovados
                            │   │   │       │           ├─ Sim: FOLHA 338 | previsão: APROVADO (1) | bootstrap: 1 recusados, 9 aprovados
                            │   │   │       │           └─ Não: FOLHA 339 | previsão: APROVADO (1) | bootstrap: 2 recusados, 8 aprovados
                            │   │   │       └─ Não: FOLHA 342 | previsão: APROVADO (1) | bootstrap: 0 recusados, 94 aprovados
                            │   │   └─ Não: NÓ 330 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 7 recusados, 96 aprovados
                            │   │       ├─ Sim: FOLHA 332 | previsão: APROVADO (1) | bootstrap: 7 recusados, 81 aprovados
                            │   │       └─ Não: FOLHA 331 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                            │   └─ Não: NÓ 266 | debt_to_income_ratio = '38'? | bootstrap: 259 recusados, 4400 aprovados
                            │       ├─ Sim: NÓ 320 | derived_race = 'Joint'? | bootstrap: 5 recusados, 172 aprovados
                            │       │   ├─ Sim: FOLHA 328 | previsão: APROVADO (1) | bootstrap: 2 recusados, 25 aprovados
                            │       │   └─ Não: NÓ 321 | property_value (valor do imóvel em USD) <= 220000.0? | bootstrap: 3 recusados, 147 aprovados
                            │       │       ├─ Sim: FOLHA 322 | previsão: APROVADO (1) | bootstrap: 0 recusados, 55 aprovados
                            │       │       └─ Não: NÓ 323 | income (renda anual declarada em milhares de USD) <= 74.0? | bootstrap: 3 recusados, 92 aprovados
                            │       │           ├─ Sim: FOLHA 324 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
                            │       │           └─ Não: NÓ 325 | derived_race = 'Black or African American'? | bootstrap: 1 recusados, 80 aprovados
                            │       │               ├─ Sim: FOLHA 327 | previsão: APROVADO (1) | bootstrap: 0 recusados, 54 aprovados
                            │       │               └─ Não: FOLHA 326 | previsão: APROVADO (1) | bootstrap: 1 recusados, 26 aprovados
                            │       └─ Não: NÓ 267 | applicant_age = '55-64'? | bootstrap: 254 recusados, 4228 aprovados
                            │           ├─ Sim: NÓ 315 | debt_to_income_ratio = '40'? | bootstrap: 48 recusados, 631 aprovados
                            │           │   ├─ Sim: FOLHA 319 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
                            │           │   └─ Não: NÓ 316 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 48 recusados, 603 aprovados
                            │           │       ├─ Sim: FOLHA 318 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                            │           │       └─ Não: FOLHA 317 | previsão: APROVADO (1) | bootstrap: 48 recusados, 596 aprovados
                            │           └─ Não: NÓ 268 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 206 recusados, 3597 aprovados
                            │               ├─ Sim: FOLHA 314 | previsão: APROVADO (1) | bootstrap: 1 recusados, 71 aprovados
                            │               └─ Não: NÓ 269 | debt_to_income_ratio = '46'? | bootstrap: 205 recusados, 3526 aprovados
                            │                   ├─ Sim: NÓ 305 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 1 recusados, 188 aprovados
                            │                   │   ├─ Sim: NÓ 307 | applicant_age = '<25'? | bootstrap: 1 recusados, 160 aprovados
                            │                   │   │   ├─ Sim: FOLHA 313 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
                            │                   │   │   └─ Não: NÓ 308 | income (renda anual declarada em milhares de USD) <= 49.0? | bootstrap: 1 recusados, 148 aprovados
                            │                   │   │       ├─ Sim: NÓ 309 | loan_amount (valor solicitado em USD) <= 170000.0? | bootstrap: 1 recusados, 16 aprovados
                            │                   │   │       │   ├─ Sim: FOLHA 310 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                            │                   │   │       │   └─ Não: FOLHA 311 | previsão: APROVADO (1) | bootstrap: 1 recusados, 7 aprovados
                            │                   │   │       └─ Não: FOLHA 312 | previsão: APROVADO (1) | bootstrap: 0 recusados, 132 aprovados
                            │                   │   └─ Não: FOLHA 306 | previsão: APROVADO (1) | bootstrap: 0 recusados, 28 aprovados
                            │                   └─ Não: NÓ 270 | construction_method = '2'? | bootstrap: 204 recusados, 3338 aprovados
                            │                       ├─ Sim: NÓ 296 | debt_to_income_ratio = '42'? | bootstrap: 9 recusados, 64 aprovados
                            │                       │   ├─ Sim: FOLHA 304 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                            │                       │   └─ Não: NÓ 297 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 9 recusados, 54 aprovados
                            │                       │       ├─ Sim: NÓ 299 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 111.5? | bootstrap: 9 recusados, 46 aprovados
                            │                       │       │   ├─ Sim: NÓ 300 | derived_race = 'Asian'? | bootstrap: 5 recusados, 42 aprovados
                            │                       │       │   │   ├─ Sim: FOLHA 302 | previsão: APROVADO (1) | bootstrap: 3 recusados, 4 aprovados
                            │                       │       │   │   └─ Não: FOLHA 301 | previsão: APROVADO (1) | bootstrap: 2 recusados, 38 aprovados
                            │                       │       │   └─ Não: FOLHA 303 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 4 aprovados
                            │                       │       └─ Não: FOLHA 298 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                            │                       └─ Não: NÓ 271 | loan_type = '1'? | bootstrap: 195 recusados, 3274 aprovados
                            │                           ├─ Sim: NÓ 293 | debt_to_income_ratio = '<20%'? | bootstrap: 13 recusados, 64 aprovados
                            │                           │   ├─ Sim: FOLHA 295 | previsão: APROVADO (1) | bootstrap: 3 recusados, 8 aprovados
                            │                           │   └─ Não: FOLHA 294 | previsão: APROVADO (1) | bootstrap: 10 recusados, 56 aprovados
                            │                           └─ Não: NÓ 272 | debt_to_income_ratio = '50%-60%'? | bootstrap: 182 recusados, 3210 aprovados
                            │                               ├─ Sim: NÓ 282 | applicant_age = '>74'? | bootstrap: 58 recusados, 1203 aprovados
                            │                               │   ├─ Sim: FOLHA 292 | previsão: APROVADO (1) | bootstrap: 2 recusados, 53 aprovados
                            │                               │   └─ Não: NÓ 283 | applicant_age = '65-74'? | bootstrap: 56 recusados, 1150 aprovados
                            │                               │       ├─ Sim: NÓ 289 | income (renda anual declarada em milhares de USD) <= 43.5? | bootstrap: 5 recusados, 153 aprovados
                            │                               │       │   ├─ Sim: FOLHA 290 | previsão: APROVADO (1) | bootstrap: 2 recusados, 10 aprovados
                            │                               │       │   └─ Não: FOLHA 291 | previsão: APROVADO (1) | bootstrap: 3 recusados, 143 aprovados
                            │                               │       └─ Não: NÓ 284 | derived_race = 'Asian'? | bootstrap: 51 recusados, 997 aprovados
                            │                               │           ├─ Sim: FOLHA 288 | previsão: APROVADO (1) | bootstrap: 3 recusados, 145 aprovados
                            │                               │           └─ Não: NÓ 285 | income (renda anual declarada em milhares de USD) <= 53.5? | bootstrap: 48 recusados, 852 aprovados
                            │                               │               ├─ Sim: FOLHA 286 | previsão: APROVADO (1) | bootstrap: 10 recusados, 33 aprovados
                            │                               │               └─ Não: FOLHA 287 | previsão: APROVADO (1) | bootstrap: 38 recusados, 819 aprovados
                            │                               └─ Não: NÓ 273 | derived_ethnicity = 'Joint'? | bootstrap: 124 recusados, 2007 aprovados
                            │                                   ├─ Sim: NÓ 275 | applicant_age = '25-34'? | bootstrap: 6 recusados, 105 aprovados
                            │                                   │   ├─ Sim: NÓ 279 | property_value (valor do imóvel em USD) <= 280000.0? | bootstrap: 4 recusados, 58 aprovados
                            │                                   │   │   ├─ Sim: FOLHA 280 | previsão: APROVADO (1) | bootstrap: 0 recusados, 24 aprovados
                            │                                   │   │   └─ Não: FOLHA 281 | previsão: APROVADO (1) | bootstrap: 4 recusados, 34 aprovados
                            │                                   │   └─ Não: NÓ 276 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 2 recusados, 47 aprovados
                            │                                   │       ├─ Sim: FOLHA 278 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
                            │                                   │       └─ Não: FOLHA 277 | previsão: APROVADO (1) | bootstrap: 2 recusados, 36 aprovados
                            │                                   └─ Não: FOLHA 274 | previsão: APROVADO (1) | bootstrap: 118 recusados, 1902 aprovados
                            └─ Não: NÓ 8 | lei = '549300KBWX4NV5Q1E376'? | bootstrap: 44255 recusados, 220610 aprovados
                                ├─ Sim: NÓ 190 | loan_term (prazo em meses) <= 270.0? | bootstrap: 921 recusados, 2643 aprovados
                                │   ├─ Sim: NÓ 191 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 74.0? | bootstrap: 4 recusados, 78 aprovados
                                │   │   ├─ Sim: FOLHA 192 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                                │   │   └─ Não: NÓ 193 | loan_term (prazo em meses) <= 210.0? | bootstrap: 2 recusados, 74 aprovados
                                │   │       ├─ Sim: NÓ 194 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 2 recusados, 67 aprovados
                                │   │       │   ├─ Sim: FOLHA 200 | previsão: APROVADO (1) | bootstrap: 0 recusados, 21 aprovados
                                │   │       │   └─ Não: NÓ 195 | derived_race = 'Asian'? | bootstrap: 2 recusados, 46 aprovados
                                │   │       │       ├─ Sim: NÓ 197 | property_value (valor do imóvel em USD) <= 480000.0? | bootstrap: 2 recusados, 36 aprovados
                                │   │       │       │   ├─ Sim: FOLHA 198 | previsão: APROVADO (1) | bootstrap: 2 recusados, 12 aprovados
                                │   │       │       │   └─ Não: FOLHA 199 | previsão: APROVADO (1) | bootstrap: 0 recusados, 24 aprovados
                                │   │       │       └─ Não: FOLHA 196 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                                │   │       └─ Não: FOLHA 201 | previsão: APROVADO (1) | bootstrap: 0 recusados, 7 aprovados
                                │   └─ Não: NÓ 202 | loan_type = '4'? | bootstrap: 917 recusados, 2565 aprovados
                                │       ├─ Sim: FOLHA 264 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 6 aprovados
                                │       └─ Não: NÓ 203 | loan_type = '1'? | bootstrap: 908 recusados, 2559 aprovados
                                │           ├─ Sim: NÓ 229 | debt_to_income_ratio = '40'? | bootstrap: 222 recusados, 1049 aprovados
                                │           │   ├─ Sim: NÓ 259 | applicant_age = '35-44'? | bootstrap: 2 recusados, 40 aprovados
                                │           │   │   ├─ Sim: NÓ 261 | derived_race = 'Asian'? | bootstrap: 2 recusados, 15 aprovados
                                │           │   │   │   ├─ Sim: FOLHA 263 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                                │           │   │   │   └─ Não: FOLHA 262 | previsão: APROVADO (1) | bootstrap: 2 recusados, 7 aprovados
                                │           │   │   └─ Não: FOLHA 260 | previsão: APROVADO (1) | bootstrap: 0 recusados, 25 aprovados
                                │           │   └─ Não: NÓ 230 | derived_ethnicity = 'Hispanic or Latino'? | bootstrap: 220 recusados, 1009 aprovados
                                │           │       ├─ Sim: FOLHA 258 | previsão: APROVADO (1) | bootstrap: 4 recusados, 19 aprovados
                                │           │       └─ Não: NÓ 231 | debt_to_income_ratio = '44'? | bootstrap: 216 recusados, 990 aprovados
                                │           │           ├─ Sim: FOLHA 257 | previsão: APROVADO (1) | bootstrap: 3 recusados, 62 aprovados
                                │           │           └─ Não: NÓ 232 | debt_to_income_ratio = '45'? | bootstrap: 213 recusados, 928 aprovados
                                │           │               ├─ Sim: FOLHA 256 | previsão: APROVADO (1) | bootstrap: 3 recusados, 55 aprovados
                                │           │               └─ Não: NÓ 233 | applicant_age = '25-34'? | bootstrap: 210 recusados, 873 aprovados
                                │           │                   ├─ Sim: NÓ 247 | derived_race = 'Joint'? | bootstrap: 62 recusados, 320 aprovados
                                │           │                   │   ├─ Sim: FOLHA 255 | previsão: APROVADO (1) | bootstrap: 0 recusados, 30 aprovados
                                │           │                   │   └─ Não: NÓ 248 | debt_to_income_ratio = '47'? | bootstrap: 62 recusados, 290 aprovados
                                │           │                   │       ├─ Sim: NÓ 252 | derived_race = 'Asian'? | bootstrap: 3 recusados, 25 aprovados
                                │           │                   │       │   ├─ Sim: FOLHA 254 | previsão: APROVADO (1) | bootstrap: 2 recusados, 13 aprovados
                                │           │                   │       │   └─ Não: FOLHA 253 | previsão: APROVADO (1) | bootstrap: 1 recusados, 12 aprovados
                                │           │                   │       └─ Não: NÓ 249 | debt_to_income_ratio = '36'? | bootstrap: 59 recusados, 265 aprovados
                                │           │                   │           ├─ Sim: FOLHA 251 | previsão: APROVADO (1) | bootstrap: 0 recusados, 15 aprovados
                                │           │                   │           └─ Não: FOLHA 250 | previsão: APROVADO (1) | bootstrap: 59 recusados, 250 aprovados
                                │           │                   └─ Não: NÓ 234 | debt_to_income_ratio = '49'? | bootstrap: 148 recusados, 553 aprovados
                                │           │                       ├─ Sim: NÓ 240 | applicant_age = '55-64'? | bootstrap: 9 recusados, 94 aprovados
                                │           │                       │   ├─ Sim: NÓ 244 | loan_amount (valor solicitado em USD) <= 330000.0? | bootstrap: 2 recusados, 12 aprovados
                                │           │                       │   │   ├─ Sim: FOLHA 245 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                                │           │                       │   │   └─ Não: FOLHA 246 | previsão: APROVADO (1) | bootstrap: 1 recusados, 6 aprovados
                                │           │                       │   └─ Não: NÓ 241 | applicant_age = '35-44'? | bootstrap: 7 recusados, 82 aprovados
                                │           │                       │       ├─ Sim: FOLHA 243 | previsão: APROVADO (1) | bootstrap: 7 recusados, 61 aprovados
                                │           │                       │       └─ Não: FOLHA 242 | previsão: APROVADO (1) | bootstrap: 0 recusados, 21 aprovados
                                │           │                       └─ Não: NÓ 235 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 139 recusados, 459 aprovados
                                │           │                           ├─ Sim: FOLHA 239 | previsão: APROVADO (1) | bootstrap: 2 recusados, 98 aprovados
                                │           │                           └─ Não: NÓ 236 | debt_to_income_ratio = '<20%'? | bootstrap: 137 recusados, 361 aprovados
                                │           │                               ├─ Sim: FOLHA 238 | previsão: APROVADO (1) | bootstrap: 2 recusados, 26 aprovados
                                │           │                               └─ Não: FOLHA 237 | previsão: APROVADO (1) | bootstrap: 135 recusados, 335 aprovados
                                │           └─ Não: NÓ 204 | debt_to_income_ratio = '38'? | bootstrap: 686 recusados, 1510 aprovados
                                │               ├─ Sim: NÓ 226 | derived_race = 'Asian'? | bootstrap: 13 recusados, 65 aprovados
                                │               │   ├─ Sim: FOLHA 228 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
                                │               │   └─ Não: FOLHA 227 | previsão: APROVADO (1) | bootstrap: 13 recusados, 53 aprovados
                                │               └─ Não: NÓ 205 | applicant_age = '<25'? | bootstrap: 673 recusados, 1445 aprovados
                                │                   ├─ Sim: FOLHA 225 | previsão: APROVADO (1) | bootstrap: 18 recusados, 58 aprovados
                                │                   └─ Não: NÓ 206 | applicant_age = '65-74'? | bootstrap: 655 recusados, 1387 aprovados
                                │                       ├─ Sim: FOLHA 224 | previsão: APROVADO (1) | bootstrap: 43 recusados, 72 aprovados
                                │                       └─ Não: NÓ 207 | derived_race = 'Asian'? | bootstrap: 612 recusados, 1315 aprovados
                                │                           ├─ Sim: NÓ 219 | debt_to_income_ratio = '44'? | bootstrap: 76 recusados, 176 aprovados
                                │                           │   ├─ Sim: NÓ 221 | income (renda anual declarada em milhares de USD) <= 115.5? | bootstrap: 3 recusados, 17 aprovados
                                │                           │   │   ├─ Sim: FOLHA 222 | previsão: APROVADO (1) | bootstrap: 0 recusados, 11 aprovados
                                │                           │   │   └─ Não: FOLHA 223 | previsão: APROVADO (1) | bootstrap: 3 recusados, 6 aprovados
                                │                           │   └─ Não: FOLHA 220 | previsão: APROVADO (1) | bootstrap: 73 recusados, 159 aprovados
                                │                           └─ Não: NÓ 208 | debt_to_income_ratio = '41'? | bootstrap: 536 recusados, 1139 aprovados
                                │                               ├─ Sim: NÓ 216 | loan_type = '2'? | bootstrap: 10 recusados, 53 aprovados
                                │                               │   ├─ Sim: FOLHA 218 | previsão: APROVADO (1) | bootstrap: 6 recusados, 34 aprovados
                                │                               │   └─ Não: FOLHA 217 | previsão: APROVADO (1) | bootstrap: 4 recusados, 19 aprovados
                                │                               └─ Não: NÓ 209 | debt_to_income_ratio = '45'? | bootstrap: 526 recusados, 1086 aprovados
                                │                                   ├─ Sim: NÓ 211 | applicant_age = '35-44'? | bootstrap: 15 recusados, 85 aprovados
                                │                                   │   ├─ Sim: FOLHA 215 | previsão: APROVADO (1) | bootstrap: 11 recusados, 39 aprovados
                                │                                   │   └─ Não: NÓ 212 | loan_amount (valor solicitado em USD) <= 330000.0? | bootstrap: 4 recusados, 46 aprovados
                                │                                   │       ├─ Sim: FOLHA 213 | previsão: APROVADO (1) | bootstrap: 4 recusados, 29 aprovados
                                │                                   │       └─ Não: FOLHA 214 | previsão: APROVADO (1) | bootstrap: 0 recusados, 17 aprovados
                                │                                   └─ Não: FOLHA 210 | previsão: APROVADO (1) | bootstrap: 511 recusados, 1001 aprovados
                                └─ Não: NÓ 9 | debt_to_income_ratio = '36'? | bootstrap: 43334 recusados, 217967 aprovados
                                    ├─ Sim: NÓ 165 | lei = '549300CB67L6KPJLHE19'? | bootstrap: 714 recusados, 7885 aprovados
                                    │   ├─ Sim: FOLHA 189 | previsão: APROVADO (1) | bootstrap: 12 recusados, 43 aprovados
                                    │   └─ Não: NÓ 166 | lei = '549300T9P3WXNMXX0I34'? | bootstrap: 702 recusados, 7842 aprovados
                                    │       ├─ Sim: FOLHA 188 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
                                    │       └─ Não: NÓ 167 | lei = '549300SK2GVCQXPD4S58'? | bootstrap: 700 recusados, 7836 aprovados
                                    │           ├─ Sim: NÓ 185 | property_value (valor do imóvel em USD) <= 355000.0? | bootstrap: 7 recusados, 9 aprovados
                                    │           │   ├─ Sim: FOLHA 186 | previsão: APROVADO (1) | bootstrap: 2 recusados, 4 aprovados
                                    │           │   └─ Não: FOLHA 187 | previsão: RECUSADO (0) | bootstrap: 5 recusados, 5 aprovados
                                    │           └─ Não: NÓ 168 | lei = '5493004AS1SPBQOFDR49'? | bootstrap: 693 recusados, 7827 aprovados
                                    │               ├─ Sim: FOLHA 184 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
                                    │               └─ Não: NÓ 169 | lei = '984500P059VACEAAFA22'? | bootstrap: 690 recusados, 7822 aprovados
                                    │                   ├─ Sim: FOLHA 183 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
                                    │                   └─ Não: NÓ 170 | lei = '54930021WPEXNHYZUL09'? | bootstrap: 687 recusados, 7817 aprovados
                                    │                       ├─ Sim: FOLHA 182 | previsão: APROVADO (1) | bootstrap: 3 recusados, 5 aprovados
                                    │                       └─ Não: NÓ 171 | applicant_age = '55-64'? | bootstrap: 684 recusados, 7812 aprovados
                                    │                           ├─ Sim: FOLHA 181 | previsão: APROVADO (1) | bootstrap: 93 recusados, 728 aprovados
                                    │                           └─ Não: NÓ 172 | lei = '254900TM81D0YC1B9584'? | bootstrap: 591 recusados, 7084 aprovados
                                    │                               ├─ Sim: FOLHA 180 | previsão: APROVADO (1) | bootstrap: 0 recusados, 24 aprovados
                                    │                               └─ Não: NÓ 173 | derived_race = 'Asian'? | bootstrap: 591 recusados, 7060 aprovados
                                    │                                   ├─ Sim: NÓ 177 | lei = '5493004N9PMBSLEZOF16'? | bootstrap: 139 recusados, 3146 aprovados
                                    │                                   │   ├─ Sim: FOLHA 179 | previsão: APROVADO (1) | bootstrap: 1 recusados, 10 aprovados
                                    │                                   │   └─ Não: FOLHA 178 | previsão: APROVADO (1) | bootstrap: 138 recusados, 3136 aprovados
                                    │                                   └─ Não: NÓ 174 | construction_method = '1'? | bootstrap: 452 recusados, 3914 aprovados
                                    │                                       ├─ Sim: FOLHA 176 | previsão: APROVADO (1) | bootstrap: 263 recusados, 3743 aprovados
                                    │                                       └─ Não: FOLHA 175 | previsão: RECUSADO (0) | bootstrap: 189 recusados, 171 aprovados
                                    └─ Não: NÓ 10 | lei = '549300W4FT4H1UWPGU95'? | bootstrap: 42620 recusados, 210082 aprovados
                                        ├─ Sim: NÓ 150 | debt_to_income_ratio = '39'? | bootstrap: 105 recusados, 245 aprovados
                                        │   ├─ Sim: FOLHA 164 | previsão: APROVADO (1) | bootstrap: 0 recusados, 18 aprovados
                                        │   └─ Não: NÓ 151 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 98.5? | bootstrap: 105 recusados, 227 aprovados
                                        │       ├─ Sim: NÓ 152 | debt_to_income_ratio = '>60%'? | bootstrap: 85 recusados, 75 aprovados
                                        │       │   ├─ Sim: FOLHA 156 | previsão: RECUSADO (0) | bootstrap: 19 recusados, 0 aprovados
                                        │       │   └─ Não: NÓ 153 | loan_type = '2'? | bootstrap: 66 recusados, 75 aprovados
                                        │       │       ├─ Sim: FOLHA 155 | previsão: APROVADO (1) | bootstrap: 5 recusados, 11 aprovados
                                        │       │       └─ Não: FOLHA 154 | previsão: APROVADO (1) | bootstrap: 61 recusados, 64 aprovados
                                        │       └─ Não: NÓ 157 | applicant_age = '55-64'? | bootstrap: 20 recusados, 152 aprovados
                                        │           ├─ Sim: FOLHA 163 | previsão: APROVADO (1) | bootstrap: 3 recusados, 9 aprovados
                                        │           └─ Não: NÓ 158 | loan_term (prazo em meses) <= 330.0? | bootstrap: 17 recusados, 143 aprovados
                                        │               ├─ Sim: FOLHA 159 | previsão: APROVADO (1) | bootstrap: 14 recusados, 32 aprovados
                                        │               └─ Não: NÓ 160 | debt_to_income_ratio = '50%-60%'? | bootstrap: 3 recusados, 111 aprovados
                                        │                   ├─ Sim: FOLHA 162 | previsão: APROVADO (1) | bootstrap: 0 recusados, 12 aprovados
                                        │                   └─ Não: FOLHA 161 | previsão: APROVADO (1) | bootstrap: 3 recusados, 99 aprovados
                                        └─ Não: NÓ 11 | loan_type = '1'? | bootstrap: 42515 recusados, 209837 aprovados
                                            ├─ Sim: NÓ 53 | lei = '5493001SXWZ4OFP8Z903'? | bootstrap: 29367 recusados, 134733 aprovados
                                            │   ├─ Sim: NÓ 117 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 127.5? | bootstrap: 259 recusados, 2568 aprovados
                                            │   │   ├─ Sim: NÓ 118 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 60.5? | bootstrap: 204 recusados, 1686 aprovados
                                            │   │   │   ├─ Sim: NÓ 119 | income (renda anual declarada em milhares de USD) <= 62.0? | bootstrap: 4 recusados, 107 aprovados
                                            │   │   │   │   ├─ Sim: FOLHA 120 | previsão: APROVADO (1) | bootstrap: 3 recusados, 7 aprovados
                                            │   │   │   │   └─ Não: FOLHA 121 | previsão: APROVADO (1) | bootstrap: 1 recusados, 100 aprovados
                                            │   │   │   └─ Não: NÓ 122 | debt_to_income_ratio = '44'? | bootstrap: 200 recusados, 1579 aprovados
                                            │   │   │       ├─ Sim: FOLHA 134 | previsão: APROVADO (1) | bootstrap: 1 recusados, 79 aprovados
                                            │   │   │       └─ Não: NÓ 123 | loan_term (prazo em meses) <= 300.0? | bootstrap: 199 recusados, 1500 aprovados
                                            │   │   │           ├─ Sim: FOLHA 124 | previsão: APROVADO (1) | bootstrap: 1 recusados, 49 aprovados
                                            │   │   │           └─ Não: NÓ 125 | debt_to_income_ratio = '45'? | bootstrap: 198 recusados, 1451 aprovados
                                            │   │   │               ├─ Sim: FOLHA 133 | previsão: APROVADO (1) | bootstrap: 3 recusados, 111 aprovados
                                            │   │   │               └─ Não: NÓ 126 | loan_amount (valor solicitado em USD) <= 480000.0? | bootstrap: 195 recusados, 1340 aprovados
                                            │   │   │                   ├─ Sim: NÓ 127 | debt_to_income_ratio = '41'? | bootstrap: 180 recusados, 1109 aprovados
                                            │   │   │                   │   ├─ Sim: FOLHA 129 | previsão: APROVADO (1) | bootstrap: 3 recusados, 72 aprovados
                                            │   │   │                   │   └─ Não: FOLHA 128 | previsão: APROVADO (1) | bootstrap: 177 recusados, 1037 aprovados
                                            │   │   │                   └─ Não: NÓ 130 | applicant_age = '55-64'? | bootstrap: 15 recusados, 231 aprovados
                                            │   │   │                       ├─ Sim: FOLHA 132 | previsão: APROVADO (1) | bootstrap: 2 recusados, 6 aprovados
                                            │   │   │                       └─ Não: FOLHA 131 | previsão: APROVADO (1) | bootstrap: 13 recusados, 225 aprovados
                                            │   │   └─ Não: NÓ 135 | derived_race = 'Asian'? | bootstrap: 55 recusados, 882 aprovados
                                            │   │       ├─ Sim: FOLHA 149 | previsão: APROVADO (1) | bootstrap: 33 recusados, 645 aprovados
                                            │   │       └─ Não: NÓ 136 | debt_to_income_ratio = '47'? | bootstrap: 22 recusados, 237 aprovados
                                            │   │           ├─ Sim: FOLHA 148 | previsão: APROVADO (1) | bootstrap: 0 recusados, 19 aprovados
                                            │   │           └─ Não: NÓ 137 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 22 recusados, 218 aprovados
                                            │   │               ├─ Sim: FOLHA 147 | previsão: APROVADO (1) | bootstrap: 0 recusados, 45 aprovados
                                            │   │               └─ Não: NÓ 138 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 131.5? | bootstrap: 22 recusados, 173 aprovados
                                            │   │                   ├─ Sim: FOLHA 139 | previsão: APROVADO (1) | bootstrap: 0 recusados, 32 aprovados
                                            │   │                   └─ Não: NÓ 140 | applicant_age = '55-64'? | bootstrap: 22 recusados, 141 aprovados
                                            │   │                       ├─ Sim: NÓ 144 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 144.5? | bootstrap: 2 recusados, 22 aprovados
                                            │   │                       │   ├─ Sim: FOLHA 145 | previsão: APROVADO (1) | bootstrap: 2 recusados, 9 aprovados
                                            │   │                       │   └─ Não: FOLHA 146 | previsão: APROVADO (1) | bootstrap: 0 recusados, 13 aprovados
                                            │   │                       └─ Não: NÓ 141 | derived_ethnicity = 'Joint'? | bootstrap: 20 recusados, 119 aprovados
                                            │   │                           ├─ Sim: FOLHA 143 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                                            │   │                           └─ Não: FOLHA 142 | previsão: APROVADO (1) | bootstrap: 20 recusados, 109 aprovados
                                            │   └─ Não: NÓ 54 | lei = '549300KTR1JGH3K6LU06'? | bootstrap: 29108 recusados, 132165 aprovados
                                            │       ├─ Sim: FOLHA 116 | previsão: APROVADO (1) | bootstrap: 30 recusados, 44 aprovados
                                            │       └─ Não: NÓ 55 | loan_amount (valor solicitado em USD) <= 170000.0? | bootstrap: 29078 recusados, 132121 aprovados
                                            │           ├─ Sim: NÓ 56 | debt_to_income_ratio = '40'? | bootstrap: 14527 recusados, 22262 aprovados
                                            │           │   ├─ Sim: NÓ 76 | lei = '7H6GLXDRUGQFU57RNE97'? | bootstrap: 338 recusados, 1179 aprovados
                                            │           │   │   ├─ Sim: FOLHA 84 | previsão: APROVADO (1) | bootstrap: 7 recusados, 18 aprovados
                                            │           │   │   └─ Não: NÓ 77 | lien_status = '1'? | bootstrap: 331 recusados, 1161 aprovados
                                            │           │   │       ├─ Sim: NÓ 79 | derived_ethnicity = 'Joint'? | bootstrap: 272 recusados, 708 aprovados
                                            │           │   │       │   ├─ Sim: NÓ 81 | construction_method = '2'? | bootstrap: 5 recusados, 33 aprovados
                                            │           │   │       │   │   ├─ Sim: FOLHA 83 | previsão: APROVADO (1) | bootstrap: 4 recusados, 18 aprovados
                                            │           │   │       │   │   └─ Não: FOLHA 82 | previsão: APROVADO (1) | bootstrap: 1 recusados, 15 aprovados
                                            │           │   │       │   └─ Não: FOLHA 80 | previsão: APROVADO (1) | bootstrap: 267 recusados, 675 aprovados
                                            │           │   │       └─ Não: FOLHA 78 | previsão: APROVADO (1) | bootstrap: 59 recusados, 453 aprovados
                                            │           │   └─ Não: NÓ 57 | lei = '5493001GDRY0EL7VG372'? | bootstrap: 14189 recusados, 21083 aprovados
                                            │           │       ├─ Sim: NÓ 73 | applicant_age = '55-64'? | bootstrap: 3 recusados, 50 aprovados
                                            │           │       │   ├─ Sim: FOLHA 75 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                                            │           │       │   └─ Não: FOLHA 74 | previsão: APROVADO (1) | bootstrap: 3 recusados, 42 aprovados
                                            │           │       └─ Não: NÓ 58 | lei = '5493009XSE00M8B9R434'? | bootstrap: 14186 recusados, 21033 aprovados
                                            │           │           ├─ Sim: NÓ 66 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 107 recusados, 15 aprovados
                                            │           │           │   ├─ Sim: NÓ 70 | loan_amount (valor solicitado em USD) <= 110000.0? | bootstrap: 16 recusados, 2 aprovados
                                            │           │           │   │   ├─ Sim: FOLHA 71 | previsão: RECUSADO (0) | bootstrap: 12 recusados, 0 aprovados
                                            │           │           │   │   └─ Não: FOLHA 72 | previsão: RECUSADO (0) | bootstrap: 4 recusados, 2 aprovados
                                            │           │           │   └─ Não: NÓ 67 | applicant_age = '65-74'? | bootstrap: 91 recusados, 13 aprovados
                                            │           │           │       ├─ Sim: FOLHA 69 | previsão: RECUSADO (0) | bootstrap: 14 recusados, 0 aprovados
                                            │           │           │       └─ Não: FOLHA 68 | previsão: RECUSADO (0) | bootstrap: 77 recusados, 13 aprovados
                                            │           │           └─ Não: NÓ 59 | debt_to_income_ratio = '20%-<30%'? | bootstrap: 14079 recusados, 21018 aprovados
                                            │           │               ├─ Sim: NÓ 63 | lei = '549300XQVJ1XBNFA5536'? | bootstrap: 1859 recusados, 4118 aprovados
                                            │           │               │   ├─ Sim: FOLHA 65 | previsão: RECUSADO (0) | bootstrap: 353 recusados, 317 aprovados
                                            │           │               │   └─ Não: FOLHA 64 | previsão: APROVADO (1) | bootstrap: 1506 recusados, 3801 aprovados
                                            │           │               └─ Não: NÓ 60 | income (renda anual declarada em milhares de USD) <= 38.5? | bootstrap: 12220 recusados, 16900 aprovados
                                            │           │                   ├─ Sim: FOLHA 61 | previsão: RECUSADO (0) | bootstrap: 4614 recusados, 2306 aprovados
                                            │           │                   └─ Não: FOLHA 62 | previsão: APROVADO (1) | bootstrap: 7606 recusados, 14594 aprovados
                                            │           └─ Não: NÓ 85 | construction_method = '2'? | bootstrap: 14551 recusados, 109859 aprovados
                                            │               ├─ Sim: NÓ 109 | lei = '549300VZVN841I2ILS84'? | bootstrap: 1849 recusados, 824 aprovados
                                            │               │   ├─ Sim: FOLHA 115 | previsão: APROVADO (1) | bootstrap: 1 recusados, 11 aprovados
                                            │               │   └─ Não: NÓ 110 | lei = '549300FGXN1K3HLB1R50'? | bootstrap: 1848 recusados, 813 aprovados
                                            │               │       ├─ Sim: FOLHA 114 | previsão: APROVADO (1) | bootstrap: 8 recusados, 12 aprovados
                                            │               │       └─ Não: NÓ 111 | total_units = '2'? | bootstrap: 1840 recusados, 801 aprovados
                                            │               │           ├─ Sim: FOLHA 113 | previsão: APROVADO (1) | bootstrap: 0 recusados, 9 aprovados
                                            │               │           └─ Não: FOLHA 112 | previsão: RECUSADO (0) | bootstrap: 1840 recusados, 792 aprovados
                                            │               └─ Não: NÓ 86 | lei = '549300121SF0K2LN2804'? | bootstrap: 12702 recusados, 109035 aprovados
                                            │                   ├─ Sim: NÓ 100 | derived_race = 'Native Hawaiian or Other Pacific Islander'? | bootstrap: 56 recusados, 890 aprovados
                                            │                   │   ├─ Sim: FOLHA 108 | previsão: APROVADO (1) | bootstrap: 0 recusados, 8 aprovados
                                            │                   │   └─ Não: NÓ 101 | derived_ethnicity = 'Not Hispanic or Latino'? | bootstrap: 56 recusados, 882 aprovados
                                            │                   │       ├─ Sim: NÓ 105 | property_value (valor do imóvel em USD) <= 650000.0? | bootstrap: 51 recusados, 816 aprovados
                                            │                   │       │   ├─ Sim: FOLHA 106 | previsão: APROVADO (1) | bootstrap: 48 recusados, 554 aprovados
                                            │                   │       │   └─ Não: FOLHA 107 | previsão: APROVADO (1) | bootstrap: 3 recusados, 262 aprovados
                                            │                   │       └─ Não: NÓ 102 | derived_race = 'Joint'? | bootstrap: 5 recusados, 66 aprovados
                                            │                   │           ├─ Sim: FOLHA 104 | previsão: APROVADO (1) | bootstrap: 3 recusados, 18 aprovados
                                            │                   │           └─ Não: FOLHA 103 | previsão: APROVADO (1) | bootstrap: 2 recusados, 48 aprovados
                                            │                   └─ Não: NÓ 87 | income (renda anual declarada em milhares de USD) <= 44.5? | bootstrap: 12646 recusados, 108145 aprovados
                                            │                       ├─ Sim: NÓ 88 | lei = '549300FNXYY540N23N64'? | bootstrap: 1369 recusados, 1746 aprovados
                                            │                       │   ├─ Sim: FOLHA 92 | previsão: APROVADO (1) | bootstrap: 26 recusados, 43 aprovados
                                            │                       │   └─ Não: NÓ 89 | loan_term (prazo em meses) <= 330.0? | bootstrap: 1343 recusados, 1703 aprovados
                                            │                       │       ├─ Sim: FOLHA 90 | previsão: RECUSADO (0) | bootstrap: 90 recusados, 26 aprovados
                                            │                       │       └─ Não: FOLHA 91 | previsão: APROVADO (1) | bootstrap: 1253 recusados, 1677 aprovados
                                            │                       └─ Não: NÓ 93 | lien_status = '1'? | bootstrap: 11277 recusados, 106399 aprovados
                                            │                           ├─ Sim: NÓ 97 | debt_to_income_ratio = '49'? | bootstrap: 10673 recusados, 105786 aprovados
                                            │                           │   ├─ Sim: FOLHA 99 | previsão: APROVADO (1) | bootstrap: 755 recusados, 12187 aprovados
                                            │                           │   └─ Não: FOLHA 98 | previsão: APROVADO (1) | bootstrap: 9918 recusados, 93599 aprovados
                                            │                           └─ Não: NÓ 94 | tract_to_msa_income_percentage (renda relativa do setor em porcentagem) <= 102.5? | bootstrap: 604 recusados, 613 aprovados
                                            │                               ├─ Sim: FOLHA 95 | previsão: RECUSADO (0) | bootstrap: 192 recusados, 89 aprovados
                                            │                               └─ Não: FOLHA 96 | previsão: APROVADO (1) | bootstrap: 412 recusados, 524 aprovados
                                            └─ Não: NÓ 12 | lei = '549300TDRENNEZ2S6M03'? | bootstrap: 13148 recusados, 75104 aprovados
                                                ├─ Sim: FOLHA 52 | previsão: APROVADO (1) | bootstrap: 0 recusados, 192 aprovados
                                                └─ Não: NÓ 13 | lei = '5493007L4GYNO2PUBS09'? | bootstrap: 13148 recusados, 74912 aprovados
                                                    ├─ Sim: FOLHA 51 | previsão: APROVADO (1) | bootstrap: 15 recusados, 19 aprovados
                                                    └─ Não: NÓ 14 | derived_ethnicity = 'Joint'? | bootstrap: 13133 recusados, 74893 aprovados
                                                        ├─ Sim: NÓ 32 | lei = 'WWB2V0FCW3A0EE3ZJN75'? | bootstrap: 582 recusados, 4475 aprovados
                                                        │   ├─ Sim: FOLHA 50 | previsão: APROVADO (1) | bootstrap: 4 recusados, 7 aprovados
                                                        │   └─ Não: NÓ 33 | loan_term (prazo em meses) <= 304.0? | bootstrap: 578 recusados, 4468 aprovados
                                                        │       ├─ Sim: NÓ 34 | loan_type = '2'? | bootstrap: 19 recusados, 20 aprovados
                                                        │       │   ├─ Sim: FOLHA 36 | previsão: RECUSADO (0) | bootstrap: 15 recusados, 12 aprovados
                                                        │       │   └─ Não: FOLHA 35 | previsão: APROVADO (1) | bootstrap: 4 recusados, 8 aprovados
                                                        │       └─ Não: NÓ 37 | property_value (valor do imóvel em USD) <= 240000.0? | bootstrap: 559 recusados, 4448 aprovados
                                                        │           ├─ Sim: NÓ 38 | debt_to_income_ratio = '<20%'? | bootstrap: 92 recusados, 416 aprovados
                                                        │           │   ├─ Sim: FOLHA 42 | previsão: APROVADO (1) | bootstrap: 4 recusados, 14 aprovados
                                                        │           │   └─ Não: NÓ 39 | derived_race = 'American Indian or Alaska Native'? | bootstrap: 88 recusados, 402 aprovados
                                                        │           │       ├─ Sim: FOLHA 41 | previsão: APROVADO (1) | bootstrap: 3 recusados, 32 aprovados
                                                        │           │       └─ Não: FOLHA 40 | previsão: APROVADO (1) | bootstrap: 85 recusados, 370 aprovados
                                                        │           └─ Não: NÓ 43 | lei = '5493004WMLN60ZJ2ON46'? | bootstrap: 467 recusados, 4032 aprovados
                                                        │               ├─ Sim: NÓ 47 | debt_to_income_ratio = '>60%'? | bootstrap: 12 recusados, 66 aprovados
                                                        │               │   ├─ Sim: FOLHA 49 | previsão: RECUSADO (0) | bootstrap: 8 recusados, 4 aprovados
                                                        │               │   └─ Não: FOLHA 48 | previsão: APROVADO (1) | bootstrap: 4 recusados, 62 aprovados
                                                        │               └─ Não: NÓ 44 | debt_to_income_ratio = '>60%'? | bootstrap: 455 recusados, 3966 aprovados
                                                        │                   ├─ Sim: FOLHA 46 | previsão: RECUSADO (0) | bootstrap: 193 recusados, 105 aprovados
                                                        │                   └─ Não: FOLHA 45 | previsão: APROVADO (1) | bootstrap: 262 recusados, 3861 aprovados
                                                        └─ Não: NÓ 15 | lei = '5493008OE5JXNTGWD962'? | bootstrap: 12551 recusados, 70418 aprovados
                                                            ├─ Sim: FOLHA 31 | previsão: APROVADO (1) | bootstrap: 16 recusados, 34 aprovados
                                                            └─ Não: NÓ 16 | lei = '549300GPO6DWUZR4UY30'? | bootstrap: 12535 recusados, 70384 aprovados
                                                                ├─ Sim: FOLHA 30 | previsão: APROVADO (1) | bootstrap: 18 recusados, 198 aprovados
                                                                └─ Não: NÓ 17 | lei = '549300AQ3T62GXDU7D76'? | bootstrap: 12517 recusados, 70186 aprovados
                                                                    ├─ Sim: NÓ 25 | loan_term (prazo em meses) <= 330.0? | bootstrap: 129 recusados, 1774 aprovados
                                                                    │   ├─ Sim: FOLHA 26 | previsão: APROVADO (1) | bootstrap: 0 recusados, 10 aprovados
                                                                    │   └─ Não: NÓ 27 | debt_to_income_ratio = '39'? | bootstrap: 129 recusados, 1764 aprovados
                                                                    │       ├─ Sim: FOLHA 29 | previsão: APROVADO (1) | bootstrap: 0 recusados, 57 aprovados
                                                                    │       └─ Não: FOLHA 28 | previsão: APROVADO (1) | bootstrap: 129 recusados, 1707 aprovados
                                                                    └─ Não: NÓ 18 | lei = '7H6GLXDRUGQFU57RNE97'? | bootstrap: 12388 recusados, 68412 aprovados
                                                                        ├─ Sim: NÓ 22 | loan_amount (valor solicitado em USD) <= 110000.0? | bootstrap: 115 recusados, 164 aprovados
                                                                        │   ├─ Sim: FOLHA 23 | previsão: RECUSADO (0) | bootstrap: 9 recusados, 1 aprovados
                                                                        │   └─ Não: FOLHA 24 | previsão: APROVADO (1) | bootstrap: 106 recusados, 163 aprovados
                                                                        └─ Não: NÓ 19 | lei = '549300AG64NHILB7ZP05'? | bootstrap: 12273 recusados, 68248 aprovados
                                                                            ├─ Sim: FOLHA 21 | previsão: APROVADO (1) | bootstrap: 554 recusados, 1686 aprovados
                                                                            └─ Não: FOLHA 20 | previsão: APROVADO (1) | bootstrap: 11719 recusados, 66562 aprovados
```
