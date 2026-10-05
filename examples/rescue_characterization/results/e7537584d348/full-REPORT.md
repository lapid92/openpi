# Frozen π₀.₅ rescue characterization

Complete planned paired evaluations; no training or adaptive step policy.

Each benchmark is reported separately. LIBERO and LIBERO-Plus use the same frozen LIBERO checkpoint; no RoboCasa episodes were evaluated.

Checkpoint full content SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Historical metadata identity: dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8.

## Benchmark results

### libero

Unresolved: too few one-step failures or rescues under the declared budget.

Benchmark revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c. 800 paired cases, 40 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/yviwny1x)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 768/800 (96.00%) | — | — | — |
| 2 | 771/800 (96.38%) | 17 / 14 | 0.38% [-1.12%, 1.75%] | 17/32 (53.12%) |
| 4 | 769/800 (96.12%) | 18 / 17 | 0.12% [-1.25%, 1.50%] | 18/32 (56.25%) |
| 10 | 778/800 (97.25%) | 20 / 10 | 1.25% [-0.12%, 3.00%] | 20/32 (62.50%) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 26147 | 32.90 / 33.61 | 49.09 / 89.08 |
| 2 | 2 | 51604 | 34.90 / 35.64 | 48.72 / 87.19 |
| 4 | 4 | 104496 | 38.92 / 39.67 | 49.24 / 88.10 |
| 10 | 10 | 258610 | 51.04 / 51.84 | 49.33 / 88.17 |

#### First-success and non-monotonic patterns

Full success-pattern counts: {"fail/fail/fail/fail": 8, "fail/fail/fail/succeed": 4, "fail/fail/succeed/fail": 1, "fail/fail/succeed/succeed": 2, "fail/succeed/fail/fail": 1, "fail/succeed/fail/succeed": 1, "fail/succeed/succeed/fail": 2, "fail/succeed/succeed/succeed": 13, "succeed/fail/fail/fail": 2, "succeed/fail/fail/succeed": 3, "succeed/fail/succeed/fail": 1, "succeed/fail/succeed/succeed": 8, "succeed/succeed/fail/fail": 1, "succeed/succeed/fail/succeed": 11, "succeed/succeed/succeed/fail": 6, "succeed/succeed/succeed/succeed": 736}
First success among one-step failures: {"10": 4, "2": 17, "4": 3, "never": 8}
One-step failures: 32.
Primary uncertainty uses whole base families for Plus and tasks for LIBERO. Full intervals and distributions are in libero-patterns.json.
Visual failure labels remain pending independent review; numerical rescues alone do not establish a failure mechanism.

#### Task-family results

**KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 18/20 (90.00%) | 1 / 2 | -5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 18/20 (90.00%) | 1 / 2 | -5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**KITCHEN_SCENE8_put_both_moka_pots_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/20 (50.00%) | — | — | — |
| 2 | 11/20 (55.00%) | 2 / 1 | 5.00% unavailable (<2 clusters) | 2/10 (20.00%) |
| 4 | 11/20 (55.00%) | 3 / 2 | 5.00% unavailable (<2 clusters) | 3/10 (30.00%) |
| 10 | 15/20 (75.00%) | 6 / 1 | 25.00% unavailable (<2 clusters) | 6/10 (60.00%) |

**LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 18/20 (90.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 2 / 0 | 10.00% unavailable (<2 clusters) | 2/2 (100.00%) |
| 4 | 20/20 (100.00%) | 2 / 0 | 10.00% unavailable (<2 clusters) | 2/2 (100.00%) |
| 10 | 20/20 (100.00%) | 2 / 0 | 10.00% unavailable (<2 clusters) | 2/2 (100.00%) |

**LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 18/20 (90.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 4 | 18/20 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 10 | 18/20 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/2 (50.00%) |

**LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |

**LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 18/20 (90.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 4 | 16/20 (80.00%) | 0 / 2 | -10.00% unavailable (<2 clusters) | 0/2 (0.00%) |
| 10 | 18/20 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/2 (50.00%) |

**STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 18/20 (90.00%) | — | — | — |
| 2 | 17/20 (85.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/2 (0.00%) |
| 4 | 17/20 (85.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/2 (0.00%) |
| 10 | 17/20 (85.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/2 (0.00%) |

**open_the_middle_drawer_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**open_the_top_drawer_and_put_the_bowl_inside**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 17/20 (85.00%) | 0 / 3 | -15.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_alphabet_soup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_bbq_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 18/20 (90.00%) | 0 / 2 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_butter_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_chocolate_pudding_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**pick_up_the_cream_cheese_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**pick_up_the_ketchup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_milk_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_orange_juice_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 10 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |

**pick_up_the_salad_dressing_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_tomato_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 18/20 (90.00%) | 0 / 2 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**push_the_plate_to_the_front_of_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_bowl_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_bowl_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 10 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |

**put_the_bowl_on_top_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_cream_cheese_in_the_bowl**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 19/20 (95.00%) | 0 / 1 | -5.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_wine_bottle_on_the_rack**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 19/20 (95.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**put_the_wine_bottle_on_top_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 18/20 (90.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 2 / 0 | 10.00% unavailable (<2 clusters) | 2/2 (100.00%) |
| 4 | 20/20 (100.00%) | 2 / 0 | 10.00% unavailable (<2 clusters) | 2/2 (100.00%) |
| 10 | 19/20 (95.00%) | 1 / 0 | 5.00% unavailable (<2 clusters) | 1/2 (50.00%) |

**turn_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

### libero_plus

Paired rescue and regression cases observed; interpret net differences with clustered uncertainty.

Benchmark revision: 4976dc30028e805ff8094b55501d532c48fec182. 1680 paired cases, 168 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-rescue-characterization/runs/t917y67i)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 1531/1680 (91.13%) | — | — | — |
| 2 | 1536/1680 (91.43%) | 42 / 37 | 0.30% [-0.71%, 1.37%] | 42/149 (28.19%) |
| 4 | 1534/1680 (91.31%) | 53 / 50 | 0.18% [-0.95%, 1.31%] | 53/149 (35.57%) |
| 10 | 1529/1680 (91.01%) | 53 / 55 | -0.12% [-1.25%, 1.13%] | 53/149 (35.57%) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 60170 | 32.90 / 33.62 | 61.85 / 121.09 |
| 2 | 2 | 120100 | 34.92 / 35.63 | 61.82 / 125.77 |
| 4 | 4 | 240220 | 38.93 / 39.67 | 61.93 / 119.00 |
| 10 | 10 | 605340 | 51.04 / 51.84 | 62.94 / 121.49 |

#### First-success and non-monotonic patterns

Full success-pattern counts: {"fail/fail/fail/fail": 80, "fail/fail/fail/succeed": 4, "fail/fail/succeed/fail": 7, "fail/fail/succeed/succeed": 16, "fail/succeed/fail/fail": 7, "fail/succeed/fail/succeed": 5, "fail/succeed/succeed/fail": 2, "fail/succeed/succeed/succeed": 28, "succeed/fail/fail/fail": 6, "succeed/fail/fail/succeed": 7, "succeed/fail/succeed/fail": 2, "succeed/fail/succeed/succeed": 22, "succeed/succeed/fail/fail": 14, "succeed/succeed/fail/succeed": 23, "succeed/succeed/succeed/fail": 33, "succeed/succeed/succeed/succeed": 1424}
First success among one-step failures: {"10": 4, "2": 42, "4": 23, "never": 80}
One-step failures: 149.
Primary uncertainty uses whole base families for Plus and tasks for LIBERO. Full intervals and distributions are in libero_plus-patterns.json.
Visual failure labels remain pending independent review; numerical rescues alone do not establish a failure mechanism.

#### Task-family results

**KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 105/120 (87.50%) | — | — | — |
| 2 | 112/120 (93.33%) | 8 / 1 | 5.83% [0.83%, 11.67%] | 8/15 (53.33%) |
| 4 | 109/120 (90.83%) | 6 / 2 | 3.33% [-0.83%, 7.50%] | 6/15 (40.00%) |
| 10 | 111/120 (92.50%) | 8 / 2 | 5.00% [0.83%, 9.17%] | 8/15 (53.33%) |

**KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 56/60 (93.33%) | — | — | — |
| 2 | 54/60 (90.00%) | 1 / 3 | -3.33% [-6.67%, 0.00%] | 1/4 (25.00%) |
| 4 | 55/60 (91.67%) | 3 / 4 | -1.67% [-8.33%, 6.67%] | 3/4 (75.00%) |
| 10 | 58/60 (96.67%) | 3 / 1 | 3.33% [-5.00%, 15.00%] | 3/4 (75.00%) |

**KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 39/40 (97.50%) | — | — | — |
| 2 | 37/40 (92.50%) | 1 / 3 | -5.00% [-22.50%, 7.50%] | 1/1 (100.00%) |
| 4 | 38/40 (95.00%) | 1 / 2 | -2.50% [-10.00%, 5.00%] | 1/1 (100.00%) |
| 10 | 36/40 (90.00%) | 0 / 3 | -7.50% [-15.00%, 0.00%] | 0/1 (0.00%) |

**LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 94/100 (94.00%) | — | — | — |
| 2 | 91/100 (91.00%) | 3 / 6 | -3.00% [-7.00%, 2.00%] | 3/6 (50.00%) |
| 4 | 93/100 (93.00%) | 2 / 3 | -1.00% [-4.00%, 2.00%] | 2/6 (33.33%) |
| 10 | 94/100 (94.00%) | 2 / 2 | 0.00% [-3.00%, 3.00%] | 2/6 (33.33%) |

**LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 68/80 (85.00%) | — | — | — |
| 2 | 69/80 (86.25%) | 2 / 1 | 1.25% [0.00%, 3.75%] | 2/12 (16.67%) |
| 4 | 72/80 (90.00%) | 6 / 2 | 5.00% [1.25%, 10.00%] | 6/12 (50.00%) |
| 10 | 69/80 (86.25%) | 4 / 3 | 1.25% [-2.50%, 5.00%] | 4/12 (33.33%) |

**LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**open_the_middle_drawer_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 125/140 (89.29%) | — | — | — |
| 2 | 130/140 (92.86%) | 6 / 1 | 3.57% [-0.71%, 8.57%] | 6/15 (40.00%) |
| 4 | 128/140 (91.43%) | 5 / 2 | 2.14% [-2.86%, 9.29%] | 5/15 (33.33%) |
| 10 | 127/140 (90.71%) | 7 / 5 | 1.43% [-4.29%, 8.57%] | 7/15 (46.67%) |

**open_the_top_drawer_and_put_the_bowl_inside**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 62/80 (77.50%) | — | — | — |
| 2 | 61/80 (76.25%) | 2 / 3 | -1.25% [-5.00%, 2.50%] | 2/18 (11.11%) |
| 4 | 61/80 (76.25%) | 3 / 4 | -1.25% [-7.50%, 3.75%] | 3/18 (16.67%) |
| 10 | 61/80 (76.25%) | 3 / 4 | -1.25% [-7.50%, 3.75%] | 3/18 (16.67%) |

**pick_up_the_alphabet_soup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 106/120 (88.33%) | — | — | — |
| 2 | 103/120 (85.83%) | 4 / 7 | -2.50% [-6.67%, 1.67%] | 4/14 (28.57%) |
| 4 | 107/120 (89.17%) | 4 / 3 | 0.83% [-3.33%, 5.00%] | 4/14 (28.57%) |
| 10 | 109/120 (90.83%) | 4 / 1 | 2.50% [-0.83%, 6.67%] | 4/14 (28.57%) |

**pick_up_the_bbq_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 59/60 (98.33%) | — | — | — |
| 2 | 59/60 (98.33%) | 1 / 1 | 0.00% [-5.00%, 5.00%] | 1/1 (100.00%) |
| 4 | 58/60 (96.67%) | 1 / 2 | -1.67% [-6.67%, 3.33%] | 1/1 (100.00%) |
| 10 | 60/60 (100.00%) | 1 / 0 | 1.67% [0.00%, 5.00%] | 1/1 (100.00%) |

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 109/110 (99.09%) | — | — | — |
| 2 | 109/110 (99.09%) | 1 / 1 | 0.00% [0.00%, 0.00%] | 1/1 (100.00%) |
| 4 | 110/110 (100.00%) | 1 / 0 | 0.91% [0.00%, 2.73%] | 1/1 (100.00%) |
| 10 | 110/110 (100.00%) | 1 / 0 | 0.91% [0.00%, 2.73%] | 1/1 (100.00%) |

**pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 119/120 (99.17%) | — | — | — |
| 2 | 120/120 (100.00%) | 1 / 0 | 0.83% [0.00%, 2.50%] | 1/1 (100.00%) |
| 4 | 119/120 (99.17%) | 1 / 1 | 0.00% [0.00%, 0.00%] | 1/1 (100.00%) |
| 10 | 119/120 (99.17%) | 1 / 1 | 0.00% [0.00%, 0.00%] | 1/1 (100.00%) |

**pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 64/70 (91.43%) | — | — | — |
| 2 | 64/70 (91.43%) | 2 / 2 | 0.00% [-4.29%, 4.29%] | 2/6 (33.33%) |
| 4 | 63/70 (90.00%) | 2 / 3 | -1.43% [-5.71%, 2.86%] | 2/6 (33.33%) |
| 10 | 61/70 (87.14%) | 3 / 6 | -4.29% [-8.57%, 1.43%] | 3/6 (50.00%) |

**pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 88/90 (97.78%) | — | — | — |
| 2 | 88/90 (97.78%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/2 (0.00%) |
| 4 | 87/90 (96.67%) | 1 / 2 | -1.11% [-3.33%, 0.00%] | 1/2 (50.00%) |
| 10 | 87/90 (96.67%) | 0 / 1 | -1.11% [-3.33%, 0.00%] | 0/2 (0.00%) |

**pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 9/10 (90.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 10 | 9/10 (90.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |

**pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 8/10 (80.00%) | 0 / 2 | -20.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 8/10 (80.00%) | 0 / 2 | -20.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 7/10 (70.00%) | — | — | — |
| 2 | 8/10 (80.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/3 (33.33%) |
| 4 | 7/10 (70.00%) | 3 / 3 | 0.00% unavailable (<2 clusters) | 3/3 (100.00%) |
| 10 | 6/10 (60.00%) | 2 / 3 | -10.00% unavailable (<2 clusters) | 2/3 (66.67%) |

**pick_up_the_butter_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 59/60 (98.33%) | — | — | — |
| 2 | 60/60 (100.00%) | 1 / 0 | 1.67% [0.00%, 5.00%] | 1/1 (100.00%) |
| 4 | 58/60 (96.67%) | 1 / 2 | -1.67% [-6.67%, 3.33%] | 1/1 (100.00%) |
| 10 | 59/60 (98.33%) | 1 / 1 | 0.00% [-5.00%, 5.00%] | 1/1 (100.00%) |

**pick_up_the_cream_cheese_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 62/70 (88.57%) | — | — | — |
| 2 | 63/70 (90.00%) | 2 / 1 | 1.43% [-2.86%, 5.71%] | 2/8 (25.00%) |
| 4 | 61/70 (87.14%) | 3 / 4 | -1.43% [-8.57%, 4.29%] | 3/8 (37.50%) |
| 10 | 62/70 (88.57%) | 3 / 3 | 0.00% [-4.29%, 4.29%] | 3/8 (37.50%) |

**pick_up_the_ketchup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**pick_up_the_milk_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_salad_dressing_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 74/80 (92.50%) | — | — | — |
| 2 | 74/80 (92.50%) | 1 / 1 | 0.00% [-3.75%, 3.75%] | 1/6 (16.67%) |
| 4 | 73/80 (91.25%) | 0 / 1 | -1.25% [-3.75%, 0.00%] | 0/6 (0.00%) |
| 10 | 73/80 (91.25%) | 0 / 1 | -1.25% [-3.75%, 0.00%] | 0/6 (0.00%) |

**pick_up_the_tomato_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 7/10 (70.00%) | — | — | — |
| 2 | 6/10 (60.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/3 (0.00%) |
| 4 | 6/10 (60.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/3 (0.00%) |
| 10 | 5/10 (50.00%) | 0 / 2 | -20.00% unavailable (<2 clusters) | 0/3 (0.00%) |

**push_the_plate_to_the_front_of_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 46/60 (76.67%) | — | — | — |
| 2 | 46/60 (76.67%) | 1 / 1 | 0.00% [-5.00%, 5.00%] | 1/14 (7.14%) |
| 4 | 48/60 (80.00%) | 4 / 2 | 3.33% [0.00%, 6.67%] | 4/14 (28.57%) |
| 10 | 43/60 (71.67%) | 3 / 6 | -5.00% [-13.33%, 1.67%] | 3/14 (21.43%) |

**put_the_bowl_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 58/60 (96.67%) | — | — | — |
| 2 | 59/60 (98.33%) | 1 / 0 | 1.67% [0.00%, 5.00%] | 1/2 (50.00%) |
| 4 | 59/60 (98.33%) | 1 / 0 | 1.67% [0.00%, 5.00%] | 1/2 (50.00%) |
| 10 | 57/60 (95.00%) | 1 / 2 | -1.67% [-10.00%, 5.00%] | 1/2 (50.00%) |

**put_the_wine_bottle_on_top_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 67/80 (83.75%) | — | — | — |
| 2 | 65/80 (81.25%) | 1 / 3 | -2.50% [-8.75%, 2.50%] | 1/13 (7.69%) |
| 4 | 66/80 (82.50%) | 3 / 4 | -1.25% [-11.25%, 6.25%] | 3/13 (23.08%) |
| 10 | 66/80 (82.50%) | 4 / 5 | -1.25% [-8.75%, 5.00%] | 4/13 (30.77%) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

## Cost and uncertainty boundaries

Policy latency includes synchronized production inference and input/output transforms; it excludes Gaussian noise generation, JSON and networking. Episode time includes simulator construction, reset, stabilization, policy requests and simulation; it excludes teardown. Compilation and parity are separate preparation records.
Arm tables use condition-cluster intervals. The pattern JSON additionally gives the PRIMARY Plus family-cluster intervals. Use these for scientific conclusions; condition clustering is sensitivity. Zero discordance does not prove equivalence.
Objects Layout supplies one state per condition repeated at ten seeds. Plus uses severity3/4 across7 axes with10 seeds; standard LIBERO uses20 seeds. Conditions are a predeclared subset, not the official full benchmark.
Fewer than50 one-step failures fails the predeclared adequacy target. A descriptive recurring class requires at least10 rescues across3 families and5 conditions in the new population, with counterexamples and uncertainty. No adaptive speedup is claimed.

## Reproduction

Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.
~~~bash
setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase smoke > /volt/artifacts/rescue-characterization/smoke-supervisor.log 2>&1
setsid -f .venv/bin/python examples/rescue_characterization/supervise.py --manifest examples/rescue_characterization/protocol.json --output /volt/artifacts/rescue-characterization/runs --phase full > /volt/artifacts/rescue-characterization/full-supervisor.log 2>&1
~~~

Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.
