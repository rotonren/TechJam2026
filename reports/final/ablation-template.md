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
| Selected v2 system | Dev folds 1-4 | - | - | - | 0.519195 mean |
| Selected v2 system | Sealed fold 5, once | 0.600 | 0.365893 | 5.75 | 0.514768 |
| Selected v2 system | Full public set | 0.625 | 0.321365 | 5.53 | 0.518309 |
| Optimized candidate | Dev folds 1-4 | - | - | - | 0.662377 mean |
| Optimized candidate, dense reproduction | Full public set | 0.840 | 0.376349 | 4.615 | 0.660605 |
| Lexical-only operations control | Full public set | 0.865 | 0.464843 | 4.38 | 0.704353 |

Selected configuration: commit `4c41adf`, tag `compasscart-v2-candidate`, config hash
`0b6bcec8230b618c71604d2770b49b83495845d59f9ae452e11a607b7127373c`,
selection score `0.500756`, runtime API cost USD 0.00.

Optimized candidate runtime: commit `54b2a62`, proposed post-review tag
`compasscart-v3-candidate`, canonical config hash
`4400c69e62a123979d7cadef7b9384f975b9fda28dd18aa93a8a3758544b65e5`, and
folds 1-4 selection score `0.644160`. The lexical-only row is a public-set
operations control and was not used to change the selected default dense
runtime.
