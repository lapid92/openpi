# Frozen π₀.₅ fixed flow-step study

Complete planned paired evaluations; no training or adaptive step policy.

Each benchmark is reported separately. LIBERO and LIBERO-Plus use the same frozen LIBERO checkpoint; no RoboCasa episodes were evaluated.

Checkpoint full content SHA256: 9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5.
Historical metadata identity: dad4e2fbe79cceca79b83f3e53bb59c180e55ebb6815c67bfcdcf81768d7cee8.

## Benchmark results

### libero

Unresolved: too few one-step failures or rescues under the declared budget.

Benchmark revision: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c. 400 paired cases, 40 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-frozen-flow-study/runs/opu92158)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 386/400 (96.50%) | — | — | — |
| 2 | 390/400 (97.50%) | 8 / 4 | 1.00% [-0.50%, 2.50%] | 8/14 (57.14%) |
| 4 | 390/400 (97.50%) | 11 / 7 | 1.00% [-1.00%, 3.00%] | 11/14 (78.57%) |
| 10 | 386/400 (96.50%) | 11 / 11 | 0.00% [-2.00%, 2.00%] | 11/14 (78.57%) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 12880 | 33.02 / 34.02 | 45.86 / 80.22 |
| 2 | 2 | 25616 | 35.02 / 36.03 | 45.72 / 80.26 |
| 4 | 4 | 51196 | 39.02 / 40.05 | 45.82 / 79.15 |
| 10 | 10 | 128870 | 51.01 / 52.07 | 46.30 / 78.48 |

#### Task-family results

**KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**KITCHEN_SCENE8_put_both_moka_pots_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 6/10 (60.00%) | — | — | — |
| 2 | 7/10 (70.00%) | 2 / 1 | 10.00% unavailable (<2 clusters) | 2/4 (50.00%) |
| 4 | 4/10 (40.00%) | 2 / 4 | -20.00% unavailable (<2 clusters) | 2/4 (50.00%) |
| 10 | 4/10 (40.00%) | 2 / 4 | -20.00% unavailable (<2 clusters) | 2/4 (50.00%) |

**LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**open_the_middle_drawer_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**open_the_top_drawer_and_put_the_bowl_inside**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 8/10 (80.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 4 | 9/10 (90.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 10 | 9/10 (90.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/2 (50.00%) |

**pick_up_the_alphabet_soup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_bbq_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_butter_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_chocolate_pudding_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_cream_cheese_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_ketchup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_milk_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_orange_juice_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_salad_dressing_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_tomato_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**push_the_plate_to_the_front_of_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 9/10 (90.00%) | 1 / 1 | 0.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**put_the_bowl_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_bowl_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 9/10 (90.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/1 (0.00%) |
| 4 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |
| 10 | 10/10 (100.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/1 (100.00%) |

**put_the_bowl_on_top_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_cream_cheese_in_the_bowl**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**put_the_wine_bottle_on_the_rack**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 8/10 (80.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 1 / 0 | 10.00% unavailable (<2 clusters) | 1/2 (50.00%) |
| 4 | 10/10 (100.00%) | 2 / 0 | 20.00% unavailable (<2 clusters) | 2/2 (100.00%) |
| 10 | 10/10 (100.00%) | 2 / 0 | 20.00% unavailable (<2 clusters) | 2/2 (100.00%) |

**put_the_wine_bottle_on_top_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**turn_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

### libero_plus

Paired rescue and regression cases observed; interpret net differences with clustered uncertainty.

Benchmark revision: 4976dc30028e805ff8094b55501d532c48fec182. 480 paired cases, 48 task/condition clusters.
[W&B](https://wandb.ai/arm-aair-idit/pi05-frozen-flow-study/runs/egcqq28j)

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 454/480 (94.58%) | — | — | — |
| 2 | 456/480 (95.00%) | 15 / 13 | 0.42% [-2.08%, 3.12%] | 15/26 (57.69%) |
| 4 | 459/480 (95.62%) | 16 / 11 | 1.04% [-1.25%, 3.33%] | 16/26 (61.54%) |
| 10 | 463/480 (96.46%) | 16 / 7 | 1.88% [-0.42%, 4.38%] | 16/26 (61.54%) |

| Steps | Velocity evaluations/chunk | Total velocity evaluations | Policy call mean / p95 (ms) | Episode mean / p95 (s) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 16608 | 33.00 / 34.04 | 49.80 / 85.20 |
| 2 | 2 | 33060 | 34.96 / 36.01 | 49.60 / 85.84 |
| 4 | 4 | 65376 | 39.01 / 40.10 | 49.39 / 82.14 |
| 10 | 10 | 161560 | 50.99 / 52.07 | 49.21 / 84.19 |

#### Task-family results

**KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 15/20 (75.00%) | — | — | — |
| 2 | 16/20 (80.00%) | 4 / 3 | 5.00% [-20.00%, 30.00%] | 4/5 (80.00%) |
| 4 | 18/20 (90.00%) | 4 / 1 | 15.00% [10.00%, 20.00%] | 4/5 (80.00%) |
| 10 | 20/20 (100.00%) | 5 / 0 | 25.00% [10.00%, 40.00%] | 5/5 (100.00%) |

**KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

**LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 35/40 (87.50%) | — | — | — |
| 2 | 34/40 (85.00%) | 0 / 1 | -2.50% [-7.50%, 0.00%] | 0/5 (0.00%) |
| 4 | 36/40 (90.00%) | 2 / 1 | 2.50% [-7.50%, 15.00%] | 2/5 (40.00%) |
| 10 | 34/40 (85.00%) | 1 / 2 | -2.50% [-15.00%, 7.50%] | 1/5 (20.00%) |

**LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 38/40 (95.00%) | — | — | — |
| 2 | 40/40 (100.00%) | 2 / 0 | 5.00% [0.00%, 15.00%] | 2/2 (100.00%) |
| 4 | 39/40 (97.50%) | 2 / 1 | 2.50% [0.00%, 7.50%] | 2/2 (100.00%) |
| 10 | 39/40 (97.50%) | 2 / 1 | 2.50% [0.00%, 7.50%] | 2/2 (100.00%) |

**open_the_middle_drawer_of_the_cabinet**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 58/60 (96.67%) | — | — | — |
| 2 | 59/60 (98.33%) | 1 / 0 | 1.67% [0.00%, 5.00%] | 1/2 (50.00%) |
| 4 | 57/60 (95.00%) | 0 / 1 | -1.67% [-5.00%, 0.00%] | 0/2 (0.00%) |
| 10 | 58/60 (96.67%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/2 (0.00%) |

**open_the_top_drawer_and_put_the_bowl_inside**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 19/20 (95.00%) | — | — | — |
| 2 | 20/20 (100.00%) | 1 / 0 | 5.00% [0.00%, 10.00%] | 1/1 (100.00%) |
| 4 | 20/20 (100.00%) | 1 / 0 | 5.00% [0.00%, 10.00%] | 1/1 (100.00%) |
| 10 | 20/20 (100.00%) | 1 / 0 | 5.00% [0.00%, 10.00%] | 1/1 (100.00%) |

**pick_up_the_alphabet_soup_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 58/60 (96.67%) | — | — | — |
| 2 | 59/60 (98.33%) | 2 / 1 | 1.67% [-5.00%, 10.00%] | 2/2 (100.00%) |
| 4 | 59/60 (98.33%) | 2 / 1 | 1.67% [-5.00%, 10.00%] | 2/2 (100.00%) |
| 10 | 60/60 (100.00%) | 2 / 0 | 3.33% [0.00%, 10.00%] | 2/2 (100.00%) |

**pick_up_the_bbq_sauce_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 20/20 (100.00%) | — | — | — |
| 2 | 19/20 (95.00%) | 0 / 1 | -5.00% [-10.00%, 0.00%] | 0/0 (undefined) |
| 4 | 20/20 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 20/20 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

**pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 49/50 (98.00%) | — | — | — |
| 2 | 49/50 (98.00%) | 1 / 1 | 0.00% [-6.00%, 6.00%] | 1/1 (100.00%) |
| 4 | 50/50 (100.00%) | 1 / 0 | 2.00% [0.00%, 6.00%] | 1/1 (100.00%) |
| 10 | 49/50 (98.00%) | 1 / 1 | 0.00% [-6.00%, 6.00%] | 1/1 (100.00%) |

**pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 10/10 (100.00%) | 0 / 0 | 0.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 40/40 (100.00%) | — | — | — |
| 2 | 40/40 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 4 | 40/40 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |
| 10 | 40/40 (100.00%) | 0 / 0 | 0.00% [0.00%, 0.00%] | 0/0 (undefined) |

**pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 10/10 (100.00%) | — | — | — |
| 2 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 4 | 8/10 (80.00%) | 0 / 2 | -20.00% unavailable (<2 clusters) | 0/0 (undefined) |
| 10 | 9/10 (90.00%) | 0 / 1 | -10.00% unavailable (<2 clusters) | 0/0 (undefined) |

**pick_up_the_cream_cheese_and_place_it_in_the_basket**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 32/40 (80.00%) | — | — | — |
| 2 | 34/40 (85.00%) | 4 / 2 | 5.00% [-5.00%, 15.00%] | 4/8 (50.00%) |
| 4 | 34/40 (85.00%) | 4 / 2 | 5.00% [-5.00%, 15.00%] | 4/8 (50.00%) |
| 10 | 35/40 (87.50%) | 4 / 1 | 7.50% [0.00%, 15.00%] | 4/8 (50.00%) |

**put_the_bowl_on_the_stove**

| Steps | Success | Rescues / regressions vs 1 | Net difference [95% cluster interval] | One-step failures rescued |
|---:|---:|---:|---:|---:|
| 1 | 40/40 (100.00%) | — | — | — |
| 2 | 37/40 (92.50%) | 0 / 3 | -7.50% [-15.00%, 0.00%] | 0/0 (undefined) |
| 4 | 38/40 (95.00%) | 0 / 2 | -5.00% [-15.00%, 0.00%] | 0/0 (undefined) |
| 10 | 39/40 (97.50%) | 0 / 1 | -2.50% [-7.50%, 0.00%] | 0/0 (undefined) |

Per-family costs, condition outcomes, and every paired case are in the JSON summary. Raw JSONL preserves chunk records and error attempts.

## Cost and uncertainty boundaries

Policy latency includes synchronized production inference and input/output transforms; it excludes Gaussian noise generation, JSON and networking. Episode time includes simulator construction, reset, stabilization, policy requests and simulation; it excludes teardown. Compilation and parity are separate preparation records.
Intervals use 10,000 condition-cluster bootstrap draws retaining all states/arms. Plus conditions may share a base family, so condition clustering can understate family dependence. Single-condition family intervals are unavailable; zero discordance does not prove equivalence.
Objects Layout supplies one state per condition repeated at ten seeds. Other selected conditions use ten state indices. Severity-2/3 Plus conditions are an exploratory subset, not the official full benchmark.
Fewer than 20 one-step failures or five rescues leaves rescue benefit unresolved. No adaptive speedup is claimed.

## RoboCasa blocker

Public candidate changyeon/pi05_robocasa_as50_jax at revision 165da7e92fdbd140c4fe3e2a4bad6f0cabcda47e contains a completed Orbax save. Its configuration, camera/state/action adapter, and simulator provenance could not be verified. It was not evaluated with the LIBERO interface. See setup/ROBOCASA_DISCOVERY.md.

## Reproduction

Execution directory: /volt/code/frozen-flow-study on Volt pod uz2ptakxucbe. The manifest pins full checkpoint files, source files, benchmark assets, tasks, states, seeds and budgets. Status JSON records exact child argument lists and W&B links.
~~~bash
setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase smoke > /volt/artifacts/frozen-flow-study/smoke-supervisor.log 2>&1
setsid -f .venv/bin/python examples/frozen_flow/supervise.py --manifest examples/frozen_flow/protocol.json --output /volt/artifacts/frozen-flow-study/runs --phase full > /volt/artifacts/frozen-flow-study/full-supervisor.log 2>&1
~~~

Errors halt execution and remain on the pod; publication occurs only after complete audits. Publication failures are recorded in supervisor status and require publication retry, not episode replacement.
