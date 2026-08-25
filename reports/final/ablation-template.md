# CompassCart Ablation Record

The early rows use the full public set where available; later iteration rows use
development fold 1 and are compared only with other fold-1 rows. Final fold 1-4
CV is reported separately to avoid implying that unlike scopes are equivalent.

| System | Evaluation scope | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | --- | ---: | ---: | ---: | ---: |
| Official starter BM25 | Full public set | 0.125 | 0.068034 | 9.81 | 0.106710 |
| CompassCart lexical | Full public set | 0.355 | 0.229875 | 8.67 | 0.293063 |
| + local ONNX dense | Dev fold 1 | - | - | - | 0.318342 |
| + override pending-state fix | Dev fold 1 | - | - | - | 0.335342 |
| + bounded versioned query history | Dev fold 1 | 0.500 | 0.182946 | 7.30 | 0.378884 |
| + answerable clarification policy | Dev fold 1 | 0.575 | 0.194673 | 6.05 | 0.444902 |
| Selected full system | Dev folds 1-4 | - | - | - | 0.424038 mean |

Selected configuration: commit `dc71a7a`, config hash
`0b6bcec8230b618c71604d2770b49b83495845d59f9ae452e11a607b7127373c`,
selection score `0.405810`, runtime API cost USD 0.00.
