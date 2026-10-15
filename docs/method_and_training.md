# Method and Training

## Equations and Modules

| Paper equations | Computation | Implementation |
| --- | --- | --- |
| (1)-(2) | Multi-scale convolution, temporal average pooling, MLP, and autoregressive base forecast | MTM in `models.py` |
| (3) | Relation cross-entropy supervised by three-teacher majority labels | `experts.py` and `training.py` |
| (4) | Softmax of the raw option logits; the keep probability is the edge weight | `decision.py` |
| (5) | Probability-weighted source aggregation divided by the candidate count | TGM in `models.py` |
| (6)-(7) | Bounded graph correction and spatiotemporal forecast | TGM in `models.py` |
| (8) | Forecasting MAE for joint optimization of theta and phi | `training.py` |
| (9)-(10) | Bounded target-region calibration and final forecast | CM in `models.py` |

MTM uses a 20-week window and five convolution branches with eight channels per branch, producing a 40-dimensional temporal representation. The graph head takes `[h_target, message]`; CM takes `h_target`. Their hidden dimensions default to the temporal representation dimension.

[configs/default.json](../configs/default.json) records reference method settings and is not loaded automatically. Callers must read it and pass the values to the corresponding model, data, and training interfaces, constructing each `StageConfig` with its stage settings. `StageConfig()` uses its own constructor defaults.

## Text Context and Relation Supervision

`SummaryInput` contains the target and source histories in the same observation window and case scale, together with the source-to-target importation pressure. The pressure is supplied from `float(candidate.pressure[target, source])`: inbound passenger-flow share multiplied by the sum of the source's raw case rates per 10,000 people over the two latest complete weeks.

`generate_summary` uses the supplied summary provider to produce four nonempty fields in order: `Target`, `Source`, `Comparison`, and `Importation pressure`. The resulting summary S is shared by the teacher models and the decision model. The pressure field records the direction and supplied score.

`PairContext` combines the region pair, forecast origin, horizon, summary S, and available event text E. Three distinct teacher providers receive the same prompt and return a JSON string, `{"label": 0}` or `{"label": 1}`. At least two retain votes produce label 1; otherwise, the label is 0. Malformed responses raise an error.

Teacher labels are stored in `RelationBatch.labels` for relation supervision. Laya produces two raw option logits ordered as `reject=0` and `keep=1`. Their softmax keep probability provides the continuous edge weight. During forecasting, the current context determines this probability.

## Source Selection and Forecasting

Matrices use `[target, source]` order. For each forecast origin, air connectivity uses the twelve complete months preceding the release month. The passenger-flow denominator includes all observed inbound sources and is retained after selecting up to ten candidates. Eligible sources have positive passenger flow and two known weekly case rates; zero case rates are valid. Ties are ordered by descending passenger flow and ascending source index.

TGM weights source temporal representations by retention probability and divides the aggregate by the candidate count. A message projection and the target representation produce the bounded graph correction. An empty candidate set yields zero graph correction.

CM uses the target-region temporal representation to compute a bounded correction. It is trained with the current decision model and forecaster fixed, minimizing final-forecast MAE to learn residual-bias correction.

## Training Stages

Here, theta denotes Laya's `head`, `type_emb`, and `scorer` parameters. The other pretrained Laya parameters remain fixed. Phi comprises MTM and the graph forecasting components.

| Call / stage | Optimized parameters | Objective | Fixed parameters |
| --- | --- | --- | --- |
| `pretrain_relations` | theta | Relation cross-entropy | Other Laya parameters, phi, CM |
| `initialize` / core | phi | Spatiotemporal-forecast MAE | theta_0, CM |
| `initialize` / joint | theta, phi | Spatiotemporal-forecast MAE | Other Laya parameters, CM |
| `initialize` / calibration | Calibration head and alpha_L | Final-forecast MAE | theta, phi |
| `update_month` / core | phi | Spatiotemporal-forecast MAE | theta, CM |
| `update_month` / calibration | Calibration head and alpha_L | Final-forecast MAE | theta, phi |
| `predict` | None | None | All model parameters |

Each stage uses its own optimizer, validation-based early stopping, and restoration of the best parameters. Joint optimization includes the initial state as a candidate and restores theta and phi from the same selected state.

Monthly updates inherit existing parameters, first update the forecaster, then fix it to update CM. Model parameters remain fixed within the month; weekly context produces the edge weights. An interrupted initialization or update restores the previous model and scheduling state.

## Time and Scale Conventions

- `forecast_origin` is the end date of the latest complete observation week. `target_week = forecast_origin + 7 * horizon` identifies the single target week. Each horizon is trained separately.
- The release date is `forecast_origin + 1 day`; monthly scheduling and the air-flow window use the release month.
- Summaries use observations available through the forecast origin, with explicit case units and scale. Missing observations are marked `unknown`.
- Event text records its regional scope and `available_at`, optionally with `effective_at`. Only text available by the forecast origin is included. When no event text is available, the context uses `no event text available`.
- Training and validation target weeks must be strictly earlier than `cutoff`. Their sample identities and forecast origins are checked for overlap. `ForecastBatch.contexts` follows the candidate mask's batch/target/source order.

The forecasting scale is `ln(1+x)` of weekly new cases per 10,000 people, followed by per-region min-max normalization fitted on the initial training period. The default fitting period is 2020. Constant regions use a normalization span of one; subsequent observations are not clipped. Candidate selection uses raw case rates per 10,000 people.

Evaluation inverse-normalizes forecasts to the log1p scale, computes RMSE and MAE separately for each region, and averages them with equal region weights, as implemented in `data.py`.
