# Frozen π₀.₅ rescue characterization — final research report
Completed fixed-sampler study and blinded qualitative review. No predictor, adaptive rule, TVM, training, or adaptive speedup was used.

The independently declared scan is complete: 800 LIBERO cases and 1,680 LIBERO-Plus cases, each evaluated at fixed 1/2/4/10 flow steps (9,920 episodes). Additional steps rescued some one-step failures, but did not yield a monotonic success ladder. Every overall net-success interval includes zero. The evidence supports repeated paired rescue outcomes across multiple task families, but the completed visual sample does not establish a single failure class rescued by a particular tested step count under the declared recurrence criterion. Visible acquisition and placement differences occur, alongside regressions, uncertain completion boundaries, and one simulator-affected case.

## New independent scan
The fixed manifest was declared before outcomes. Standard LIBERO covers all 40 tasks in four suites, states 10–29 and seeds 3001–3020. Plus covers 168 conditions: four suites × seven axes × severity 3/4 × three distinct families per stratum, seeds 4001–4010 and states 20–29 when available (state 0 for single-state conditions). Conditions were selected by registry metadata excluding 144 earlier names, not by favorable outcomes. This covers 27 Plus base families, not the entire registry. All arms used matched initial states, chunk-indexed Gaussian noise, five executed actions per policy chunk, the same horizon and task-completion definition.

These are descriptive rates for the declared finite populations. Cluster bootstrap intervals quantify sensitivity to the sampled task/family mix; they do not establish representativeness of every LIBERO-Plus condition. Bootstrap uses 10,000 draws, retaining matched arms: 40 task clusters for LIBERO (seed 20261003), 27 base-family clusters for Plus (seed 20261004), with 168-condition sensitivity for Plus. No multiplicity-adjusted superiority or equivalence claim is made.

| Benchmark | Cases | One-step successes | One-step failures | Any rescue | Never rescued | One-step successes regressing at ≥1 arm | Non-monotonic cases |
|---|---|---|---|---|---|---|---|
| libero | 800 | 768 (96.00%) | 32 | 24 | 8 | 32 | 37 |
| libero_plus | 1680 | 1531 (91.13%) | 149 | 69 | 80 | 107 | 128 |

Among one-step failures, any tested larger count rescues 24/32 LIBERO cases (75.00%; primary cluster 95% interval [60.00, 91.18]%) and 69/149 Plus cases (46.31%; [37.58, 57.30]%). These intervals are the complements of the never-rescued bootstrap intervals; the any-arm statistic is retrospective and not an executable policy selection rule.

Non-monotonic means any tested success followed by failure at a larger tested count, including cases that already succeeded at one step. 'Never' means no success at the four tested counts, not impossibility at every sampler setting.

### First successful tested count among one-step failures
| Benchmark | First success | Count | % one-step failures | 95% cluster interval (%) | % all cases |
|---|---|---|---|---|---|
| libero | 2 | 17 | 53.12% | [34.38, 83.33] | 2.12% |
| libero | 4 | 3 | 9.38% | [0.00, 21.05] | 0.38% |
| libero | 10 | 4 | 12.50% | [0.00, 25.53] | 0.50% |
| libero | never | 8 | 25.00% | [8.82, 40.00] | 1.00% |
| libero_plus | 2 | 42 | 28.19% | [19.63, 38.78] | 2.50% |
| libero_plus | 4 | 23 | 15.44% | [9.23, 22.64] | 1.37% |
| libero_plus | 10 | 4 | 2.68% | [0.68, 4.95] | 0.24% |
| libero_plus | never | 80 | 53.69% | [42.70, 62.42] | 4.76% |

### Rescues and regressions versus one step
| Benchmark | Fixed steps | Successes / cases | Rescues / one-step failures | 95% rescue interval (%) | Regressions / one-step successes | Net Δ percentage points | 95% net interval (pp) |
|---|---|---|---|---|---|---|---|
| libero | 2 | 771/800 | 17/32 (53.12%) | [34.38, 83.33] | 14/768 (1.82%) | +0.38 | [-1.12, 1.75] |
| libero | 4 | 769/800 | 18/32 (56.25%) | [38.46, 85.71] | 17/768 (2.21%) | +0.12 | [-1.25, 1.50] |
| libero | 10 | 778/800 | 20/32 (62.50%) | [47.37, 78.95] | 10/768 (1.30%) | +1.25 | [-0.12, 2.88] |
| libero_plus | 2 | 1536/1680 | 42/149 (28.19%) | [19.63, 38.78] | 37/1531 (2.42%) | +0.30 | [-0.92, 1.54] |
| libero_plus | 4 | 1534/1680 | 53/149 (35.57%) | [28.57, 45.05] | 50/1531 (3.27%) | +0.18 | [-0.78, 1.09] |
| libero_plus | 10 | 1529/1680 | 53/149 (35.57%) | [27.71, 44.65] | 55/1531 (3.59%) | -0.12 | [-1.38, 0.96] |

LIBERO: 15 rescued cases remain successful at all larger tested counts, 5 subsequently regress, and 4 first succeed at 10 with no larger tested count.
LIBERO-Plus: 44 rescued cases remain successful at all larger tested counts, 21 subsequently regress, and 4 first succeed at 10 with no larger tested count.

### Cluster sensitivity and adequacy

The declared minimum of 50 one-step failures is met by Plus (149), but not LIBERO (32). LIBERO failure-subclass estimates therefore have limited precision. The 168-condition Plus sensitivity intervals for net success differences at 2/4/10 steps are [-0.71, +1.37], [-0.95, +1.37], and [-1.31, +1.13] percentage points. Corresponding rescue-rate intervals are [19.91, 38.41]%, [25.47, 48.08]%, and [25.99, 47.66]%. Family clustering remains primary because conditions share base tasks. The independent Test Writer bootstrap uses a separately implemented resampling stream and reproduces the same conclusion; small Monte Carlo interval differences are retained in its audit.

### All observed new success vectors
Order is 1/2/4/10 steps. Every pattern is retained, including reversals.
| Pattern | LIBERO | LIBERO-Plus |
|---|---|---|
| fail/fail/fail/fail | 8 | 80 |
| fail/fail/fail/succeed | 4 | 4 |
| fail/fail/succeed/fail | 1 | 7 |
| fail/fail/succeed/succeed | 2 | 16 |
| fail/succeed/fail/fail | 1 | 7 |
| fail/succeed/fail/succeed | 1 | 5 |
| fail/succeed/succeed/fail | 2 | 2 |
| fail/succeed/succeed/succeed | 13 | 28 |
| succeed/fail/fail/fail | 2 | 6 |
| succeed/fail/fail/succeed | 3 | 7 |
| succeed/fail/succeed/fail | 1 | 2 |
| succeed/fail/succeed/succeed | 8 | 22 |
| succeed/succeed/fail/fail | 1 | 14 |
| succeed/succeed/fail/succeed | 11 | 23 |
| succeed/succeed/succeed/fail | 6 | 33 |
| succeed/succeed/succeed/succeed | 736 | 1424 |

### Task, axis and severity concentration

**libero suite distribution**
| Suite | Cases | One-step failures | First 2 / 4 / 10 / never | Rescues at 2 / 4 / 10 | Regressions at 2 / 4 / 10 |
|---|---|---|---|---|---|
| libero_10 | 200 | 22 | 9 / 2 / 4 / 7 | 9 / 10 / 13 | 5 / 9 / 6 |
| libero_goal | 200 | 5 | 4 / 1 / 0 / 0 | 4 / 4 / 3 | 5 / 3 / 1 |
| libero_object | 200 | 3 | 2 / 0 / 0 / 1 | 2 / 2 / 2 | 2 / 2 / 1 |
| libero_spatial | 200 | 2 | 2 / 0 / 0 / 0 | 2 / 2 / 2 | 2 / 3 / 2 |

**libero_plus suite distribution**
| Suite | Cases | One-step failures | First 2 / 4 / 10 / never | Rescues at 2 / 4 / 10 | Regressions at 2 / 4 / 10 |
|---|---|---|---|---|---|
| libero_10 | 420 | 39 | 16 / 8 / 0 / 15 | 16 / 19 / 18 | 14 / 13 / 12 |
| libero_goal | 420 | 62 | 11 / 10 / 3 / 38 | 11 / 16 / 18 | 8 / 12 / 22 |
| libero_object | 420 | 34 | 9 / 2 / 0 / 23 | 9 / 10 / 10 | 11 / 14 / 8 |
| libero_spatial | 420 | 14 | 6 / 3 / 1 / 4 | 6 / 8 / 7 | 4 / 11 / 13 |

**LIBERO-Plus category distribution**
| category | Cases | One-step failures | First 2 / 4 / 10 / never | Rescues 2 / 4 / 10 | Regressions 2 / 4 / 10 |
|---|---|---|---|---|---|
| Background Textures | 240 | 8 | 3 / 2 / 0 / 3 | 3 / 5 / 5 | 6 / 8 / 8 |
| Camera Viewpoints | 240 | 31 | 12 / 6 / 0 / 13 | 12 / 14 / 14 | 12 / 21 / 20 |
| Language Instructions | 240 | 27 | 6 / 3 / 0 / 18 | 6 / 6 / 8 | 3 / 4 / 7 |
| Light Conditions | 240 | 11 | 3 / 4 / 1 / 3 | 3 / 7 / 5 | 9 / 6 / 6 |
| Objects Layout | 240 | 25 | 5 / 1 / 1 / 18 | 5 / 5 / 6 | 2 / 1 / 4 |
| Robot Initial States | 240 | 45 | 11 / 7 / 2 / 25 | 11 / 14 / 13 | 3 / 9 / 7 |
| Sensor Noise | 240 | 2 | 2 / 0 / 0 / 0 | 2 / 2 / 2 | 2 / 1 / 3 |

**LIBERO-Plus severity distribution**
| severity | Cases | One-step failures | First 2 / 4 / 10 / never | Rescues 2 / 4 / 10 | Regressions 2 / 4 / 10 |
|---|---|---|---|---|---|
| 3 | 840 | 62 | 18 / 9 / 1 / 34 | 18 / 21 / 21 | 16 / 20 / 26 |
| 4 | 840 | 87 | 24 / 14 / 3 / 46 | 24 / 32 / 32 | 21 / 30 / 29 |

**libero concentration and recurrence**
First success at 2: 17 cases across 14 families and 14 conditions.
First success at 4: 3 cases across 3 families and 3 conditions.
First success at 10: 4 cases across 1 families and 1 conditions.
| Family (suite) | Cases | One-step failures | Any rescue | Never | Distinct state hashes among failures |
|---|---|---|---|---|---|
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove (libero_10) | 20 | 10 | 7 | 3 | 10 |
| LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket (libero_10) | 20 | 2 | 2 | 0 | 2 |
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket (libero_10) | 20 | 2 | 1 | 1 | 2 |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate (libero_10) | 20 | 2 | 1 | 1 | 2 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy (libero_10) | 20 | 2 | 0 | 2 | 2 |
| put_the_wine_bottle_on_top_of_the_cabinet (libero_goal) | 20 | 2 | 2 | 0 | 2 |

**libero_plus concentration and recurrence**
First success at 2: 42 cases across 21 families and 34 conditions.
First success at 4: 23 cases across 12 families and 18 conditions.
First success at 10: 4 cases across 4 families and 4 conditions.
| Family (suite) | Cases | One-step failures | Any rescue | Never | Distinct state hashes among failures |
|---|---|---|---|---|---|
| open_the_top_drawer_and_put_the_bowl_inside (libero_goal) | 80 | 18 | 4 | 14 | 6 |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it (libero_10) | 120 | 15 | 10 | 5 | 10 |
| open_the_middle_drawer_of_the_cabinet (libero_goal) | 140 | 15 | 8 | 7 | 7 |
| push_the_plate_to_the_front_of_the_stove (libero_goal) | 60 | 14 | 6 | 8 | 5 |
| pick_up_the_alphabet_soup_and_place_it_in_the_basket (libero_object) | 120 | 14 | 4 | 10 | 10 |
| put_the_wine_bottle_on_top_of_the_cabinet (libero_goal) | 80 | 13 | 4 | 9 | 6 |

Ranks above are descriptive post-hoc concentration summaries, not evidence of a behavioral mechanism. Axis/severity differences confound task mix because each stratum selects families independently. Single-state Objects Layout variants vary the noise seed, not the physical initial state. Full family/condition distributions are in the pattern files; clustered overall intervals should not be copied onto these small subgroups.

### Raw-linked counterexamples
- libero, libero:libero_10:6, seed 3003, state 12: **fail/succeed/fail/succeed**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-0/part-0001.jsonl#L731), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-0/part-0001.jsonl#L732), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-0/part-0001.jsonl#L729), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-0/part-0001.jsonl#L730).
- libero, libero:libero_10:8, seed 3017, state 26: **fail/fail/succeed/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0002.jsonl#L130), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0002.jsonl#L131), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0002.jsonl#L132), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0002.jsonl#L133).
- libero, libero:libero_10:0, seed 3010, state 19: **succeed/succeed/succeed/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L600), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L597), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L598), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L599).
- libero, libero:libero_10:0, seed 3009, state 18: **fail/fail/fail/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L593), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L594), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L595), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L596).
- libero_plus, libero_plus:libero_10:1169, seed 4007, state 26: **fail/succeed/fail/succeed**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0003.jsonl#L150), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0003.jsonl#L151), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0003.jsonl#L148), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0003.jsonl#L149).
- libero_plus, libero_plus:libero_10:340, seed 4003, state 22: **fail/fail/succeed/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0003.jsonl#L48), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0003.jsonl#L49), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0003.jsonl#L46), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0003.jsonl#L47).
- libero_plus, libero_plus:libero_10:1, seed 4008, state 27: **succeed/succeed/succeed/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L500), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L501), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L502), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L499).
- libero_plus, libero_plus:libero_10:295, seed 4010, state 29: **fail/fail/fail/fail**. [1-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-3/part-0003.jsonl#L22), [2-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-3/part-0003.jsonl#L23), [4-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-3/part-0003.jsonl#L24), [10-step raw](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-3/part-0003.jsonl#L21).

The case-vector files retain task, suite, family, axis, severity, condition, seed, initial-state content hash, every arm outcome, raw file/line/content-hash references and media paths: [libero complete cases](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/libero-patterns.cases.jsonl), [libero_plus complete cases](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/libero_plus-patterns.cases.jsonl).

## Preserved simulator instability diagnostic

Four MuJoCo warnings report `Nan, Inf or huge value in QACC at DOF 9` at simulation time 0.5480. Sequential per-worker logs associate them with all four arms of exactly one declared LIBERO case: `libero:libero_10:5`, book placement into the caddy, seed 3009, initial-state index 18. All four arms ran to the 520-action horizon and were recorded as failures with `status: ok`. Thus zero episode error rows does **not** mean zero simulator warnings. Structural identity/pairing/media gates passed but did not classify this diagnostic as an execution error.

The primary analysis retains all 800 standard cases and this F/F/F/F vector. No episode was replaced or rerun. Its visual behavior cannot be attributed solely to policy failure, and the case is flagged separately in behavioral interpretation. A supplementary count excluding this single affected case gives 799 cases, 31 one-step failures, 24 any-arm rescues (77.42%), and 7 never-rescued cases; rescue counts at 2/4/10 remain 17/18/20 and regression counts remain 14/17/10. These supplementary counts do not replace the predeclared analysis or its uncertainty intervals. The warning logs, exact raw line/hash references, and reproduction script are preserved under simulator-diagnostics.

## Earlier records: descriptive, kept separate
The earlier four-arm study and independent one-versus-ten study have different declared conditions/seeds. Extraction found no duplicate physical cases across those sources. Their checkpoint and Plus benchmark revision match the new study, but the independent fixed sampler had extra head instrumentation; its timing is not pooled with the production fixed sampler. Old outcomes were not used to choose the new scan's cases.
| Earlier cohort | Cases | One-step failures | First 2 / 4 / 10 / never / unknown | Rescues 2 / 4 / 10 | Regressions 2 / 4 / 10 |
|---|---|---|---|---|---|
| frozen_flow/main/libero | 400 | 14 | 8 / 4 / 0 / 2 / 0 | 8 / 11 / 11 | 4 / 7 / 11 |
| frozen_flow/main/libero_plus | 480 | 26 | 15 / 3 / 2 / 6 / 0 | 15 / 16 / 16 | 13 / 11 / 7 |
| selective_flow/main/libero_plus | 960 | 73 | unmeasured / unmeasured / unknown / unknown / 73 | — / — / 33 | — / — / 43 |

Independent Plus has 33 ten-step rescues among 73 one-step failures; 43 one-step successes regress at ten. First success remains unknown for all 73 failures because steps 2/4 were not measured, including 40 cases failing at both 1 and 10. Those 40 must not be called never rescued at all four counts.

| Earlier cohort | Arm | Rescue % | 95% primary cluster interval (%) | Net Δ pp | 95% net interval (pp) |
|---|---|---|---|---|---|
| frozen_flow/main/libero | 2 | 57.14% | [38.46, 80.00] | +1.00 | [-0.50, 2.50] |
| frozen_flow/main/libero | 4 | 78.57% | [61.54, 100.00] | +1.00 | [-1.00, 3.00] |
| frozen_flow/main/libero | 10 | 78.57% | [61.54, 100.00] | +0.00 | [-2.00, 2.00] |
| frozen_flow/main/libero_plus | 2 | 57.69% | [31.03, 90.91] | +0.42 | [-1.82, 2.22] |
| frozen_flow/main/libero_plus | 4 | 61.54% | [43.48, 88.89] | +1.04 | [-1.14, 3.18] |
| frozen_flow/main/libero_plus | 10 | 61.54% | [35.71, 100.00] | +1.88 | [-0.42, 4.80] |
| selective_flow/main/libero_plus | 10 | 45.21% | [28.36, 67.50] | -1.04 | [-3.18, 1.40] |

Old four-arm first-success intervals among failures:
frozen_flow/main/libero: first2 8/14=57.14% [38.46, 80.00]%, first4 4/14=28.57% [9.09, 54.55]%, first10 0/14=0% [0.00, 0.00]%, never2/14=14.29% [0.00, 27.27]%. The corresponding fractions of all400cases are2.00%,1.00%,0%,0.50%. Any-arm rescue is12/14=85.71% [72.73,100.00]%, using the complement of never.
frozen_flow/main/libero_plus: first2 15/26=57.69% [31.03, 90.91]%, first4 3/26=11.54% [0.00, 25.00]%, first10 2/26=7.69% [0.00, 13.64]%, never6/26=23.08% [0.00, 42.86]%. Fractions of all480cases are3.125%,0.625%,0.417%,1.25%. Any-arm rescue is20/26=76.92% [57.14,100.00]%, using the complement of never.

Old standard has only 14 one-step failures; zero-count bootstrap intervals, such as first success at ten in that cohort, are degenerate and do not show impossibility. Old Plus has 15 family clusters; the independent cohort has 31. All intervals are exploratory. [Historical summaries and uncertainty](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/existing/summary.json); [Historical complete case catalogue](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/existing/cases.jsonl); [Historical extraction audit](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/existing/audit.json).

## Historical visual reconstruction limitation
Original older records contain action/observation hashes, not original videos or numerical action traces. The declared retrospective replay ran 548 episodes for 195 selected cases. Exactly 116 arms reproduce all recorded action/observation/noise digests, lengths and outcomes; 432 do not. Only 42 cases have all replayed arms exactly equivalent. A further independent comparison found 111 replay arms changed success outcome. A non-equivalent reconstruction cannot substantiate the original case's failure stage; agreement in outcome alone is insufficient. These replays are not new population evaluations.

The mismatch fields are action/observation hashes and downstream episode length/success, without supplied/stabilized state or noise-hash mismatch. This does not establish the cause of divergence. Investigating numerical/backend determinism is a separate task; this report must not assign causality. Exact replay pairs should be reviewed with explicit selection coverage; isolated exact arms support only single-arm observations.
[Replay exact-equivalence audit](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/historical-replay/summary.json); [Replay execution/W&B status](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/historical-replay/status.json). The status progress counter can lag the final audited 548; use completed coverage in the final audit, not a progress snapshot.

## Identity, audit and publication evidence
- Frozen manifest: pi05-rescue-characterization; SHA256 e7537584d34855c24c1a38ca11da4f7e479b9bcbfc53d2cc7ec8f7ff46eadf9d.
- Replay manifest SHA256: d9cfdf88561ec89206bd78d7104671e2d72e28cf48b12c947fc59cfd2b18cbdb.
- Checkpoint: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5 (official pi05_libero, unchanged).
- LIBERO revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c; Plus revision: 4976dc30028e805ff8094b55501d532c48fec182.
- GPUs: GPU-56335fdf-6dc8-1d64-ce39-364d4a532435, GPU-b786fcd4-f878-69e6-3017-a2ae408151fd, GPU-967ff6df-a7e2-b65a-d051-87819b7725fc, GPU-0cb759da-a716-038e-f0a9-208ec138ccc7.
- Independent full audit: 9,920 distinct expected episode records, zero duplicate/missing/unexpected/error records (four simulator warnings separately preserved below); matched initialization/noise/execution and media/action traces; 32 current benchmark×GPU×arm parity checks plus 32 prior smoke checks. Exact preparation records retain numerical differences/tolerances. Forty-four smoke episodes are excluded.
- Numerical/replay publication on branch codex/pi05-rescue-characterization is pinned at f631acc42573a5de5217cd57de4bf7c9b4bf8891. This reviewed report and its evidence package are published on the same branch; the containing Git commit identifies the final report version. Recovery amended only the previously unpublished results commit. Original ae903b8818810cdf6481e8fe0177d801a160a7cc survives in local backup ref; failure statuses/logs retained. All eight original main files and 9,920 lines reconstruct byte-for-byte from 23 repository shards. No completed simulation was rerun during publication recovery.

[Full independent gate](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/independent-final-audit.json); [Separate raw-count reproduction](audits/counts/audit.json); [Independent recovery review](audits/review-recovery.json); [Raw reconstruction index](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/index.json).

### Commands and W&B
Exact executed argument lists with timestamps and return codes are in [full run status](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/full-status.json), [recovery command ledger](recovery/state.json) and [replay ledger](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/historical-replay/status.json). Reproduction commands (do not use these to relaunch evaluations):

```bash
cd /volt/code/frozen-flow-study
.venv/bin/python examples/rescue_characterization/independent_audit.py --phase main
.venv/bin/python examples/rescue_characterization/recovery/20261005-publication/shard_records.py --index examples/rescue_characterization/results/e7537584d348/raw-shards/index.json
```

- [New scan smoke W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/ocahm4fr): 44 excluded smoke episodes.
- [Earlier four-arm W&B](https://wandb.ai/arm-aair-idit/pi05-frozen-flow-study/runs/dmxk73kz).
- [Independent historical one-versus-ten W&B](https://wandb.ai/arm-aair-idit/pi05-independent-selective-steps/runs/l83xzmjm).
- [Full evaluation W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/f1mn4gtw): original supervisor ended failed on Git publication after completed evaluation; recovery does not rewrite that W&B history.
- [Historical replay W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/591b45nq): audited replay completion.

Published immutable W&B artifacts: [smoke v0](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/artifacts/evaluation/supervisor-smoke-e7537584d348/v0), [full scan v0](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/artifacts/evaluation/supervisor-full-e7537584d348/v0), [historical replay v0](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/artifacts/retrospective-replay/historical-replay-d9cfdf88561e/v0). Links were verified through the W&B run artifact records. The original full-run artifact retains its pre-publication status; recovery provenance separately records the resolved Git packaging failure.

Declared estimate: 40–70 wall hours, 160–280 allocated H100-hours, 30–150 GiB media/traces. Main supervisor recorded 158,116.45 seconds (43.92 h) and 175.68 four-GPU allocation hours. Allocation hours are reservation accounting, not measured GPU kernel utilization. Historical replay time, audits, smoke and subsequent qualitative review are additional. No wall-clock speedup claim follows. Measured apparent on-pod directory sizes after completion: main/smoke media and traces 8,168,525,363 bytes (7.61 GiB), historical replay media/traces 680,875,862 bytes (0.63 GiB); these exclude raw JSONL, W&B copies, logs and other artifacts. The predeclared storage estimate was conservative.

## Visual review design and trace conventions

The new-case qualitative packet contains 68 cases and all 272 arms: 12 rescued, 12 regressed and 8 never-rescued LIBERO cases; 12 each for Plus. The 8 standard controls exhaust the eligible never-rescued cases. Controls are matched to selected suite/family/axis/severity strata where practical: 2 standard and 10 Plus selected controls meet the builder's matching strata. Family balancing and fixed selection seed 20261003 were declared before packet generation. Two fresh reviewers receive only opaque group/clip IDs, task instructions, success-free numeric traces and the written rubric; their initial assignments partition sorted opaque group IDs, and independent overlap is assigned without label access.

The annotators are context-separated AI reviewers, not human annotators. A third fresh reviewer covers the eligible historical packet. They receive no step counts, numeric success flags, source paths or selection mappings, but video duration and visible behavior remain outcome cues. Shared model biases and visual/semantic errors may persist across independent contexts. No human annotation or contact-force ground truth validates the stage labels; disagreement and alternatives remain part of the evidence.

Saved videos contain both external and wrist views at 20 frames per second. Frame 0 is the stabilized initial observation; frame f follows f executed actions (the last executed zero-based action index is f−1). Reported timestamps f/20 are video/simulation time, not policy wall time. Numeric EEF/gripper states precede each action and omit the final post-action state. There is no force/contact ground truth. Twelve-frame contact sheets are overview evidence; denser frames and action traces are inspected where consequential contact or placement is ambiguous. Observed completion is judged from visible evidence and may disagree with the task's numeric success predicate; both are preserved, never silently reconciled.

## Completed visual review and disagreement

All 68 selected new cases were reviewed at all four arms (272 clips). The historical packet contains all 116 eligible exact-replay clips from 46 cases: 42 cases with every originally selected arm exact and four isolated exact arms from partial cases. Only fully exact cases enter historical paired-stage claims. The 432 non-equivalent arms remain excluded, including outcome-matching reconstructions. There are no unseen clips in these packets, but most new-population cases were not selected for qualitative review.

| New benchmark / class | Available cases | Reviewed cases | Reviewed families | Reviewed conditions | Distinct initial-state hashes |
|---|---:|---:|---:|---:|---:|
| LIBERO rescue | 24 | 12 | 12 | 12 | 12 |
| LIBERO regression | 32 | 12 | 12 | 12 | 12 |
| LIBERO never at tested counts | 8 | 8 | 5 | 5 | 8 |
| Plus rescue | 69 | 12 | 12 | 12 | 12 |
| Plus regression | 107 | 12 | 12 | 12 | 12 |
| Plus never at tested counts | 80 | 12 | 5 | 7 | 11 |

The matched-control selection found 2 standard and 10 Plus cases in selected suite/family/axis/severity strata; this is not universal exact-condition matching. Remaining controls use the declared family-balanced selection. The Plus control packet includes repeated initial-state content, illustrating why seed count is not automatically state diversity.

Baseline review inspected 4,656 contact-sheet frame views and 1,103 additional frame views; independent overlap inspected 1,152 sheet views and 301 additional views. Views can overlap across sampling passes and reviewers. No reviewer claimed continuous full-video viewing. Action/EEF/gripper consultation is documented in individual coverage fields: 181 baseline and 43 overlap records contain trace-consultation narratives. These are documentation counts, not proof that absent narratives imply no trace inspection, and commands alone do not establish contact.

All 388 baseline labels were frozen by SHA256 before joining outcomes. Independent overlap contains 96 clips: 70 new and 26 historical, covering all 56 selected cohort/benchmark/family clusters and every one of the 45 initially low-confidence or unclear clips. One overlap rating was revised after the same blind reviewer inspected additional early frames; its original row, correction sidecar, rationale, timestamps and new effective export are all retained. No baseline label was rewritten.

| Independent overlap | Clips | Outcome agreement | Primary-stage agreement | Confidence agreement |
|---|---:|---:|---:|---:|
| New | 70 | 43/70 (61.43%) | 48/70 (68.57%) | 32/70 (45.71%) |
| Historical exact replay | 26 | 19/26 (73.08%) | 19/26 (73.08%) | 14/26 (53.85%) |
| Combined review sample | 96 | 62/96 (64.58%) | 67/96 (69.79%) | 46/96 (47.92%) |

Overlap intentionally oversamples uncertain clips, so these are selected-sample agreement descriptions, not population annotation-reliability estimates. There are 39 clips with a stage and/or outcome disagreement. Four direct completed-versus-not-completed conflicts received coordinator frame checks and conservative unclear/low-confidence adjudications; the other substantive differences remain explicitly unresolved. Both independent labels survive every adjudication. Typical disagreements concern whether an object at a basket rim is supported inside, whether the correct drawer was opened, and task-relative placement directions.

Across all baseline labels, confidence is high for 116 clips, medium for 228, and low for 44. Visible outcome is completed for 194, not completed for 153, and unclear for 41. Twenty-five non-unclear visual outcomes disagree with the numeric predicate (15 new, 10 historical); the numeric outcomes remain unchanged. These disagreements limit fine-grained behavioral claims. Full confusion tables, original labels, coverage, alternatives, adjudications and immutable raw links are in [review summary](review/review-summary.json), [coverage](review/coverage-details.json), [disagreement register](review/disagreement-register.json), [full labels](review/label-table.json), [compact label CSV](review/labels.csv) and [raw evidence links](review/evidence-links.json).

## Failure-class evidence at particular tested counts

For a conservative descriptive class count, the one-step arm must have a visible failure stage and the rescued arm a visible completion, with medium/high baseline confidence. Where independent overlap exists, it must agree on the relevant stage/outcome and also have medium/high confidence. An unclear adjudication excludes the pair. The simulator-affected case is excluded from policy-only class interpretation. This rule does not mean every pair received two ratings; it removes observed disagreement rather than inventing consensus for unreviewed overlaps. Alternative labels remain in the table. Counts below are selected qualitative cases, not rescue-rate estimates.

| New benchmark | Tested steps | Approach | Grasp | Manipulation | Placement | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO | 2 | 0 | 4 | 0 | 3 | 0 |
| LIBERO | 4 | 0 | 3 | 0 | 3 | 0 |
| LIBERO | 10 | 0 | 1 | 0 | 3 | 0 |
| Plus | 2 | 1 | 3 | 0 | 1 | 0 |
| Plus | 4 | 1 | 3 | 0 | 1 | 0 |
| Plus | 10 | 1 | 3 | 0 | 1 | 0 |

Each nonzero cell above spans the same number of selected families, conditions and distinct initial-state hashes as its case count. Cases can recur in multiple step columns and must not be summed as independent rescues. The largest class at a particular count is four standard grasp-stage cases at two steps; Plus has three grasp-stage cases at each tested larger count. The largest raw baseline-stage cell before the conservative screen is four cases for LIBERO and three for Plus at a particular count. Thus no reviewed new class meets the declared requirement of at least 10 rescues across three base families and five conditions.

This does not prove that such a class is absent from the unreviewed population. Only 12 of 24 standard rescued cases and 12 of 69 Plus rescued cases were selected, and label uncertainty is substantial. Broad labels such as grasp can also combine different visible events, including failed acquisition, lost retention, or interaction with the wrong object; they do not establish a single internal cause.

The numerical recurrence evidence is broader than these visual-class counts: new first-success-at-two cases span 14 standard families and 21 Plus families; first-success-at-four spans three and 12 respectively. Conversely, all four standard first-success-at-ten cases come from one moka-pot family, so that finding is concentrated in related task states rather than a broad ten-step class. The four Plus first-success-at-ten cases span four families but remain only four observations, and no larger tested count exists.

Historical exact-replay class counts remain separate. Under the same conservative visual screen, the earlier standard cohort supplies one grasp-stage case rescued at four and ten steps, and the earlier Plus cohort supplies one at four and ten. The independent historical Plus cohort supplies seven grasp-stage ten-step rescues across four families, six conditions and seven distinct state hashes, plus one placement-stage example. These are descriptive, selected reconstructions and do not satisfy the new-population confirmation rule. Their unmeasured two/four-step arms cannot identify first success.

The complete [class tables](review/characterization.json) also retain raw baseline-stage counts, regressions and controls. Historical independent controls are explicitly labelled **failed at measured 1 and 10, with 2/4 unmeasured**, preserving the builder's original selection metadata without implying never-rescued status over four counts.

## Directly checked examples and counterexamples

The coordinator inspected actual saved frames after labels were locked. These checks are unblinded supplementary observations, not additional independent ratings. Frame f is at f/20 seconds and follows action index f−1.

**New bottle acquisition and reversal.** LIBERO condition libero:libero_goal:2, seed3008, state17. The1step and10step bottle remains on the table atframe300 (15.0s). At2 and4steps it is lifted and positioned upright at the cabinet top byframe97 (4.85s). The numerical vector is F/S/S/F. Precise support/release and approach-versus-grasp cause remain alternatives. Raw records: [C090dc827d4cc4a9e19a5](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L430), [C1bec2653e5f2bd8a1fea](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L431), [C840d5738ae5cf6a31e98](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L432), [C5b01e854d990fdcb590e](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-2/part-0001.jsonl#L429).

**New placement contrast.** Plus butter/basket condition libero_plus:libero_object:81, Background Textures severity3, seed4007, state26. At1step the package stays high by the rim/hand atframes140,200,280 (7,10,14s); at2steps it is lower inside the basket with fingers apart at149 (7.45s). No contact-force inference. Raw records: [C9fe34110bc86af164805](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0001.jsonl#L467), [C09a0e9a744cec9b09a6b](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-0/part-0001.jsonl#L468).

**Matched drawer control.** Plus middle-drawer condition libero_plus:libero_goal:286, Robot Initial States severity3. Seed4008/state27 is numerically rescued at2/4/10; seed4005/state24 fails all four. Both1step clips leave fronts flush. At2steps a drawer extends in both cases, but control views show two handle/front regions above the extended drawer. Exact drawer identification and earliest failure stage remain uncertain; gross opening alone is not task success. Raw records: [C6404f5a08cf6bbde17d5](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L140), [Cb872c7c957a004c61816](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L141), [C4ad67fde7962ff97214a](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L127), [Cf422e63768b1475f0494](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-2/part-0002.jsonl#L128).

**Higher-count regression.** Plus bowl/stove condition libero_plus:libero_goal:1558, Sensor Noise severity3, seed4007, state26. The1step wrist view shows the bowl roughly horizontal over the burner at90 (4.5s). At10steps it is steeply tilted at90/150 and displaced at300. The external camera is blurred; numeric success/failure is retained separately from final support uncertainty. Raw records: [C0ace75dd78c305554e51](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-1/part-0002.jsonl#L385), [C954dcd4956421e55f879](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero_plus-main-worker-1/part-0002.jsonl#L388).

**Earlier exact reconstruction only.** Old standard wine-bottle/rack condition libero:libero_goal:9, seed1006, state5. At1step the bottle separates from the hand and lies on the table at160/191 (8.0/9.55s); at10steps it is positioned lengthwise on the rack at186/205 (9.3/10.25s). Exact available-hash equivalence permits attribution; original historical frames were never recorded. Raw records: [C8c700c642d9c969d06ec](https://github.com/lapid92/openpi/blob/e0c9d7e8b2937a2b61674dac0d90185be14e4fc9/examples/frozen_flow/results/d16dba54a815/libero-main-worker-1.jsonl#L302), [C13025af0b16d7eec66b4](https://github.com/lapid92/openpi/blob/e0c9d7e8b2937a2b61674dac0d90185be14e4fc9/examples/frozen_flow/results/d16dba54a815/libero-main-worker-1.jsonl#L301).

**Simulator-affected control.** Standard book/caddy seed3009/state18. Book and mug are visible initially but leave the external table view very early; the caddy is empty at520. The four preserved MuJoCo warnings prevent policy-only attribution. Primary population outcomes remain F/F/F/F. Raw records: [Ca38d9e6720868f30da6a](https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/examples/rescue_characterization/results/e7537584d348/raw-shards/libero-main-worker-3/part-0002.jsonl#L13).

Actual viewed frame receipts and observations are in [coordinator observations](coordinator/observations.json), [supplement](coordinator/observations-supplement.json) and the adjoining receipt files. Full videos and action arrays remain on Volt. Selected videos/contact sheets are published to W&B with the clip-to-source map; the [publication receipt](PUBLICATION.json) records immutable artifact versions and the run URL.

## Validation and delivery

The independent Test Writer reproduced all 9,920 new outcomes and 548 replay records from raw files, verified the frozen checkpoint/source identities and GPU UUIDs, and preserved every case. The full recording audit decoded all 9,920 main videos and verified traces; the replay audit verifies its media/equivalence gate. The independent Reviewer verified every visual raw link (388 original plus116 replay links), label/source hashes, the single overlap revision, all required family/low-confidence coverage, all class counts/state hashes, the four simulator-warning associations, and explicit historical measured-arm scope. Current auxiliary tests include 22 targeted tests plus 8 independent adversarial tests; detailed commands/results and final approval are under [independent review evidence](audits/final-reviewer/).

[Baseline lock](review-control/baseline-lock.json), [final overlap plan](review-control/final-blind-plan.json), [analysis commands](review-control/finalize-commands-v3.sh), [independent count reproduction](audits/counts/README.md), [independent replay reproduction](audits/replay/README.md), [diagnostic audit](simulator-diagnostics/warning-audit.json) and [publication manifest](publication-manifest.json) provide the reproduction trail. Derived versions with misleading generic historical1/10control terminology were superseded before publication; original selection metadata, original raw files and all label revisions remain preserved. No scientific condition, source-pinned sampler, or completed evaluation was changed.

## Research conclusion and limits

**The completed evidence supports repeated retrospective case-specific rescue outcomes, but does not establish a shared visual failure class rescued by a particular tested step count under the declared 10-case/3-family/5-condition criterion.** Acquisition and placement contrasts appear across several distinct families and states, so the phenomenon is not only one repeated initialization. Nevertheless, the sampled class counts are small, agreement is limited, and some apparent completions conflict with benchmark predicates. The standard first-success-at-ten result is specifically concentrated in one family.

Reject a monotonic “more steps fixes failures” interpretation: the new study contains 37 standard and 128 Plus non-monotonic cases, and higher counts introduce regressions. All overall net-success confidence intervals include zero. Do not claim overall superiority, equivalence, an online predictor, adaptive speedup, or a causal internal mechanism.

Treat the observed grasp/placement categories as hypotheses for a separately declared future characterization, not as an adaptive rule learned from these cases. Stronger class confirmation would require more independent visual evidence under a prespecified sampling plan and clearer task/contact annotation, ideally including human review. No such additional evaluations, tuning, or budget extension were performed here.
