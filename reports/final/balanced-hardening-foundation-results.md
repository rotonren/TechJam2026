# Balanced Hardening Foundation Results

Runtime commit: `f604f4758eb67b13c0e5c27de681029d90e60b27`.

## Frozen Input Integrity

All nine protected inputs matched their required SHA256 values before any
benchmark or proxy run:

| Path | SHA256 |
| --- | --- |
| `evaluator/local_evaluator.py` | `84EA899707452DE249CA62ABEE77C4B40AB7A3139B5CC798AC30C9F521F91B30` |
| `data/public_set.jsonl` | `571359A8A69014C43FC30D39C996C4A28E875DCCC249DFFC707358757BEB16C0` |
| `data/catalog.jsonl` | `DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67` |
| `assets/SHA256SUMS` | `2857869C2A872CCEA9D93BB043B8CB45EEE07CB1EFC1F943B401C1919982D86E` |
| `assets/model/model.int8.onnx` | `3013F5CDB68EA6B6A271AB8FEF96C5E6721669C2C2BE3F83EC1BE07486133892` |
| `assets/model/tokenizer.json` | `DA0E79933B9ED51798A3AE27893D3C5FA4A201126CEF75586296DF9B4D2C62A0` |
| `assets/product_vectors/product_ids.npy` | `E5AB6608C15DD0B51DD2F63DB088705613EFDFEA85859462C2D514752FE8D7C9` |
| `assets/product_vectors/scales.npy` | `3EB26371CB15A3E2AF5D287A290CD338C12C3A3F9E606BDD911C53E6D4064D53` |
| `assets/product_vectors/vectors.int8.npy` | `CCAF43034103312788DDDE27890861C6F5D93052DBC930B0B1BFF56ACF0C4D63` |

## Outside-CWD Benchmark Gate

The frozen transcript SHA256 was
`77f4399d65cabff5fcb5f0006132ba43fc47bf8571737ba6a88465ebb7066590`.
The only comparison baseline was `benchmark-r0-wall-v2.json` (SHA256
`17e32073159fef9921c791774524301b96f111e48681eb9d21d2c0548553acaf`).

The pre-R0 outside-CWD record showed the failure being addressed: dense
unavailable, response hash
`859048453631c0ca3b263c7c7976f8ed891d2bf7cbe615143c9cd752657500d3`,
P95 `321.435 ms`, and max `378.542 ms`.

All values below are milliseconds except memory values (MiB). Every R0 and P0
trial had Dense available, zero fallback, and the equivalent response hash
`29f8c08599070ba2bb57d9b8d96133b6bc9d0ddc8f28a96c606ba57cde008b00`.

| Run | Trial | Init | P50 | P95 | Max | Peak MiB | RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 1 | 13671.340 | 215.280 | 438.090 | 566.840 | 556.445 | 483.207 |
| R0 | 2 | 13676.364 | 231.930 | 555.040 | 701.676 | 555.906 | 482.664 |
| R0 | 3 | 13704.897 | 200.490 | 405.970 | 595.500 | 554.168 | 480.926 |
| P0 | 1 | 13335.379 | 126.270 | 186.680 | 289.060 | 556.453 | 483.215 |
| P0 | 2 | 13527.828 | 124.640 | 186.750 | 292.233 | 555.750 | 482.508 |
| P0 | 3 | 13479.594 | 127.860 | 189.250 | 291.380 | 556.883 | 483.641 |
| R0 median | - | 13676.364 | 215.969 | 449.975 | 701.676 | 555.906 | 482.664 |
| P0 median | - | 13479.594 | 126.076 | 186.956 | 292.233 | 556.453 | 483.215 |

P0 report SHA256: `00af02a4de3413a98c556671b72ada035b8a7dfb8524e4692d71f89c91055c17`.
The comparator reported `accepted=true`: P95 improved `58.452%`; init improved
`1.439%`; peak changed `+0.098%`; P0 maximum was below `1.5 s`. This satisfies
the material-gain and bounded-regression gate without changing the response
hash.

## Proxy Evidence

Frozen proxy manifest SHA256:
`9dfb5694a7b733952c40ea4be4661480ed8d01adb28f8fb0dc73e8f8a89f7c9d`.

| Suite | Output SHA256 | Dataset SHA256 | Fallback | Invalid | Mean technical | Selection | Std technical |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Representative dev folds 1-4 baseline | `afe5090ee8cd7b918c424d8c97dbacb744fb1a29737e931e56d010cdb24bd4cd` | `9866ffcc3836d554876d07d0350f1c8e7cabb7f26b067bf63b44f74068e9bca1` | 0 | 0 | 0.548439 | 0.543143 | 0.010594 |
| Representative dev folds 1-4 P0 | `95219696f79918a41f1ad4263594170317ade268a7321c6fde79de1c27d5e11f` | `9866ffcc3836d554876d07d0350f1c8e7cabb7f26b067bf63b44f74068e9bca1` | 0 | 0 | 0.548439 | 0.543143 | 0.010594 |
| Stress baseline | `921da182404851344ba8d51c37aee9dfd902c20395d6812ef791356aedc10d5f` | `2c8db1634fda0607fc698f3c239b0a3375b30bd08d0252497d727b5a32c67f75` | 0 | 0 | 0.596585 | 0.596585 | 0.000000 |
| Stress P0 | `109937ce1aea9c84cfc9a407d26c5f449f8dc6381c1422657018702f95114954` | `2c8db1634fda0607fc698f3c239b0a3375b30bd08d0252497d727b5a32c67f75` | 0 | 0 | 0.596585 | 0.596585 | 0.000000 |

The score, manifest, dataset, config, fallback, and invalid-response fields
are identical to the corresponding frozen development baseline. The only
non-score record difference is the explicit P0 fold selection field.

The existing audit baseline aggregate was recorded without running or reading
audit sessions: 394 samples, HitRate@10 `0.637056`, MRR `0.282756`, MTTC
`6.139594`, efficiency `0.486041`, recommended technical score `0.500563`,
fallback `0`, invalid responses `0`, and reported token usage `0`.

## Packaging And Environment

`python -m pytest -q` completed with `677 passed, 7 skipped`; Ruff on `src`,
`tests`, `tools`, and `agent.py` passed; and `git diff --check` passed.
`python -m tools.package_submission` produced a 36-file, 32.33 MiB ZIP with
SHA256 `37612efaef7abc2e0542c5541914f27ee3c9dd2c13ed4dc56ad2562820453da8`.
The submission package contract passed `4/4`, including reproducible ZIP
bytes, allowlist exclusion of evaluator/public/catalog data, and Dense loading
from an extracted ZIP under an arbitrary CWD.

Environment: Windows 11 `10.0.26200`, CPython `3.12.13`, onnxruntime `1.29.0`,
and psutil `7.2.2`. macOS verification pending.

No audit-label or final audit was run. No public evaluator or public scoring
was run.

Foundation accepted.
