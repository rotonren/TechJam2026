# Layered Attribute Schema

CompassCart keeps catalog attributes in three isolated layers. The design
borrows the useful part of a commercial attribute platform without allowing
noisy metadata to change the competition agent's hard filters accidentally.

## Layers

| Layer | Purpose | Current attributes |
|---|---|---|
| Global | Stable fields shared across product families | `category`, `brand`, `budget` |
| Category | Product-family fields | Core `material`, `color`, `size`, `style`; discovered `closure`, `fit`, `sleeve`, `neckline`, `sole_material`, `outer_material`, `inner_material`, `pattern`, `shape`, `audience` |
| Dynamic | Scenario and preference signals mined from catalog text | `feature`, `use_case`, `occasion`, `theme` |

Category fields are scoped before indexing. For example, `sole_material` and
`outer_material` are footwear fields, while `sleeve` and `neckline` are apparel
fields. A value such as `rubber sole` is therefore not treated as a replacement
for a user's whole-product material preference.

## Safety boundary

The legacy `extract_attributes` result remains the scoring-time contract used
by retrieval, hard filtering, ranking, and question selection. Layered values
are stored in `CatalogIndex.layer_inverted` and described by
`CatalogIndex.attribute_schema`.

The default parser vocabulary exposes only the existing evaluator-safe fields.
Catalog-discovered fields require explicit opt-in. This preserves current
competition behavior while leaving a controlled extension point for future
experiments or a commercial API.

## Evaluation comparison

The reference is `intent-override-opt-v2-public-2026-08-28.json`; the layered
candidate is `attribute-schema-v5-public-2026-08-29.json`.

| Metric | Previous | Layered schema | Delta |
|---|---:|---:|---:|
| HitRate@10 | 0.930000 | 0.930000 | 0.000000 |
| MRR | 0.489030 | 0.489030 | 0.000000 |
| MTTC | 3.525000 | 3.525000 | 0.000000 |
| Efficiency | 0.747500 | 0.747500 | 0.000000 |
| TechnicalScore | 0.761209 | 0.761209 | 0.000000 |

All 200 per-session outcomes are identical to the reference run.
