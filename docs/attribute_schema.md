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
by retrieval, hard filtering, ranking, and question selection.

The default parser vocabulary exposes only the existing evaluator-safe fields.
Catalog-discovered fields require explicit opt-in. This preserves current
competition behavior while leaving a controlled extension point for future
experiments or a commercial API.

## Cost of discovery, and why it is gated

`CatalogIndex` builds its schema from the flat attribute index it already
populates (`AttributeSchema.from_catalog`). Mining the catalog for the
discovered category and dynamic fields is a second pass over every product,
enabled with `CatalogIndex(..., discover_layers=True)`, which fills
`layer_inverted` and `attribute_category_scopes` and switches the schema to
`AttributeSchema.from_layers`.

Measured on the 50,000-product competition catalog:

| Path | Load | Process RSS | Parser vocabulary |
| --- | ---: | ---: | --- |
| `from_catalog` (default) | 20.72 s | +301.4 MiB | 7 fields, 21,884 values |
| `from_layers` (`discover_layers=True`) | 113.46 s | +378.8 MiB | identical |

The two vocabularies are equal field by field and value by value, because
nothing outside the discovered layer is exposed by default and no component
reads `layer_inverted`. Paying 92.7 s of startup and 77.4 MiB for an identical
result is not a trade the competition runtime should make, so discovery is off
unless a caller asks for it. Turning it on is how a future experiment would
evaluate the discovered fields.

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
