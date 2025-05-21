CARE: A Disagreement Detection Framework with Concept Alignment and Reasoning Enhancement
This repository contains the code and data for CARE: A Disagreement Detection Framework with Concept Alignment and Reasoning Enhancement.

Stages of CARE
Concept Alignment (CA)
The Concept Alignment (CA) stage is where we align different concepts.

To run this stage, execute CA_generation.py. You'll find the output in CA/data/output.

Reasoning Enhancement (RE)
The Reasoning Enhancement (RE) stage focuses on improving the reasoning capabilities of our framework. This stage involves two key phases:

Rationale to Critique: Run DS_rationale.py and DS_critique.py to generate critiques from rationales.
Counterfactual Data Augmentation: Execute CF_rewrite.py to perform counterfactual data augmentation.
Additional Utilities
Supervised Fine-Tuning (SFT)
This utility generates the necessary JSON files for Supervised Fine-Tuning (SFT).

Zero-Shot Disagreement Detection
For zero-shot disagreement detection, we utilize a taxonomy and the CARP prompt.
