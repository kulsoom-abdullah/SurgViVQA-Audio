# Data Distribution

## Dataset Overview

**Source:** [SurgViVQA Dataset](https://github.com/madratak/SurgViVQA/) - A temporally-grounded surgical VQA dataset

I used a subset of the SurgViVQA dataset (3,700 samples from 4 video IDs). Audio was generated from text questions using edge-tts to simulate spoken queries.

### Terminology Clarification
- **Video ID:** Identifier for source colonoscopy procedure (e.g., 002-001, 002-002)
- **Sample:** Individual VQA instance (question + 8 frames + answer)
- **Frames:** Temporal sequence of 8 frames extracted from video at specific timestamps

---

## Dataset Splits

| Split | Samples | Video IDs | Purpose |
|-------|---------|-----------|---------|
| **Train** | 2,302 | 002-001, 002-002, 002-003 | Model training |
| **Eval** | 398 | 002-001, 002-002, 002-003 | Validation during training |
| **Test** | 1,000 | 002-004 (held-out) | Final generalization test |
| **Total** | **3,700** | 4 colonoscopy videos | 20 question types |

---

## Reasoning Domains

Following the SurgViVQA paper structure (6 domains):

| Domain | Count | Percentage | Description |
|--------|-------|------------|-------------|
| **Instruments** | 85 | 3.7% | Tool presence, identification, counting |
| **Operation Notes** | 640 | 27.8% | Visibility, lighting, dye, occlusion |
| **Movement** | 341 | 14.8% | Scope/lesion motion, direction |
| **Other** | 1236 | 53.7% | Miscellaneous |

---

## Question Types by Answer Format

### Yes/No Questions

| Question Type | Domain | Total | Yes | No |
|---------------|--------|-------|-----|----|
| blue_dye_presence | Operation Notes | 128 | 0 | 128 |
| endoscope_visibility | Other | 128 | 44 | 84 |
| flush_action | Operation Notes | 128 | 64 | 64 |
| mucosa_visibility | Other | 128 | 66 | 62 |
| nbi_status | Operation Notes | 128 | 61 | 67 |
| occlusion_check | Other | 128 | 63 | 65 |
| scope_backward_motion | Other | 128 | 62 | 66 |
| scope_forward_motion | Movement | 128 | 63 | 65 |
| scope_motion | Movement | 128 | 65 | 63 |
| scope_outside | Other | 128 | 62 | 66 |
| tool_catheter_check | Other | 128 | 42 | 86 |

**Total Yes/No questions:** 1408 (61.2% of training set)

### Limited-Choice Questions (2-5 Options)

| Question Type | Domain | Total | Distinct Answers | Answer Distribution (Training Set) |
|---------------|--------|-------|------------------|-----------------------------------|
| **lesion_motion_direction** | Movement | 85 | **5 options** | stable (22%), left (21%), down (20%), right (20%), up (16%) |
| **lesion_screen_position** | Other | 85 | **4 options** | lower-right (27%), lower-left (25%), upper-right (25%), upper-left (24%) |
| **lesion_site** | Other | 85 | **4 options** | sigma (35%), rectum (34%), transverse (18%), descending (13%) |
| **lesion_size_range** | Other | 85 | **3 options** | <5mm (64%), 5-10mm (24%), >10mm (12%) |
| **tool_identification** | Instruments | 85 | **3 options** | forceps (65%), snare (24%), catheter (11%) |
| **fluid_occlusion_level** | Operation Notes | 128 | **3 options** | absent (47%), complete (29%), partial (24%) |
| **lighting_mode** | Operation Notes | 128 | **2 options** | WL (white light) (52%), NBI (48%) |
| **scope_motion_type** | Other | 128 | **2 options** | advancing (52%), withdrawing (48%) |
| **lesion_histology_extended** | Other | 85 | **2 options** | hyperplastic (53%), adenomatous (47%) |

**Total limited-choice questions:** 894 (38.8% of training set)

**Note:** These questions are NOT open-ended. Each has a small, fixed set of possible answers (2-5 options), making them multi-way classification tasks rather than generative QA.

---

## Overall Answer Distribution

| Answer Type | Count | Percentage |
|-------------|-------|------------|
| Yes | 592 | 25.7% |
| No | 816 | 35.4% |
| Limited-choice (2-5 options) | 894 | 38.8% |
| **Total** | **2302** | **100.0%** |

---

## Detailed Question Type Breakdown

All 20 question types across train/eval/test splits:

| Question Type | Domain | Train | Eval | Test | Total |
|---------------|--------|-------|------|------|-------|
| blue_dye_presence | Operation Notes | 128 | 22 | 50 | 200 |
| endoscope_visibility | Other | 128 | 22 | 50 | 200 |
| fluid_occlusion_level | Operation Notes | 128 | 22 | 50 | 200 |
| flush_action | Operation Notes | 128 | 22 | 50 | 200 |
| lesion_histology_extended | Other | 85 | 15 | 50 | 150 |
| lesion_motion_direction | Movement | 85 | 15 | 50 | 150 |
| lesion_screen_position | Other | 85 | 15 | 50 | 150 |
| lesion_site | Other | 85 | 15 | 50 | 150 |
| lesion_size_range | Other | 85 | 15 | 50 | 150 |
| lighting_mode | Operation Notes | 128 | 22 | 50 | 200 |
| mucosa_visibility | Other | 128 | 22 | 50 | 200 |
| nbi_status | Operation Notes | 128 | 22 | 50 | 200 |
| occlusion_check | Other | 128 | 22 | 50 | 200 |
| scope_backward_motion | Other | 128 | 22 | 50 | 200 |
| scope_forward_motion | Movement | 128 | 22 | 50 | 200 |
| scope_motion | Movement | 128 | 22 | 50 | 200 |
| scope_motion_type | Other | 128 | 22 | 50 | 200 |
| scope_outside | Other | 128 | 22 | 50 | 200 |
| tool_catheter_check | Other | 128 | 22 | 50 | 200 |
| tool_identification | Instruments | 85 | 15 | 50 | 150 |
