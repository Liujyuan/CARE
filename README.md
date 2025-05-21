# CARE: A Disagreement Detection Framework with Concept Alignment and Reasoning Enhancement

This repository contains the code and data for **CARE: A Disagreement Detection Framework with Concept Alignment and Reasoning Enhancement**.

---

## Stages of CARE

### Concept Alignment (CA)

The **Concept Alignment (CA)** stage focuses on aligning different concepts within our framework.

To run this stage, execute `CA_generation.py`. The generated output can be found in `CA/data/output`.

### Reasoning Enhancement (RE)

The **Reasoning Enhancement (RE)** stage aims to improve the reasoning capabilities of CARE. This stage involves two main phases:

1.  **Rationale to Critique:** Run `DS_rationale.py` and `DS_critique.py` to convert rationales into critiques.
2.  **Counterfactual Data Augmentation:** Execute `CF_rewrite.py` to perform counterfactual data augmentation, enhancing our dataset.

---

## Additional Utilities

### Supervised Fine-Tuning (SFT)

This utility is used to generate the necessary JSON files for **Supervised Fine-Tuning (SFT)** of our models.

### Zero-Shot Disagreement Detection

For **zero-shot disagreement detection**, we leverage a predefined taxonomy and the CARP prompt to identify disagreements without prior training on specific examples.
