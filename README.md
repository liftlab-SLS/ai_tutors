# Law Professors Prefer AI Over Peer Answers — Code & Data

This repository contains data and analysis notebooks for the paper:

> Alejandro Salinas, Carly Frieders, Neel Guha, Sibo Ma, Ralph Anzivino, Ian Ayres,  
> Oren Bar-Gill, Omri Ben-Shahar, Stephen Friedman, George Geis, Sue Guan, Christoph Henkel,  
> Stephanie Hoffer, Gregory Klass, Larasz Moody-Villarose, Sarath Sanga, Keith Sharfman,  
> Justin Simard, Rebecca Stone, David Wishnick, and Julian Nyarko.  
> *“Law Professors Prefer AI Over Peer Answers.”* Manuscript submitted for publication at  
> *Nature Human Behaviour*, 2025.

---

## Repository structure

```text
.
├── data/
│   ├── llm_as_judge_results_no_text.csv
│   └── pairwise_tutors_data_anonymized.csv
└── notebooks/
    ├── figure2_clean_full.ipynb
    ├── figure3_observed_vs_predicted_clean.ipynb
    ├── figure4_llm_as_judge_clean.ipynb
    ├── bt_ranking_D1.ipynb
    ├── win_rates_category_E1a_clean.ipynb
    ├── harmfulness_rate_F1_clean.ipynb
    ├── strength_lojo_H1H2_clean.ipynb
    ├── features_logit_figure_I1_clean.ipynb
    └── lojo_agreement_llm_judge_J1.ipynb
````

---

## Data

All data used in these notebooks is stored in the `data/` directory.

### `pairwise_tutors_data_anonymized.csv`

Pairwise-comparison data from the main experiment where judges compare answers written by:

* **Instructors (professors)**
* **LLMs** (Gemini and NotebookLM)

**Model IDs**

* **Instructor / professor answers**

  * IDs have the form:
    `prof_<anonymized_instructor_id>_q<question_id>_v<version>`
  * The anonymized instructor IDs are aligned across roles:

    * The same instructor ID is used when they act as **writer** and as **judge**.
    * Instructors are anonymized as “Instructor 1”–“Instructor 16” based on their win-rate rank in the main experiment (1 = best, 16 = worst).
  * Each instructor wrote **one** answer per question; the version suffix is not substantively meaningful for instructors.

* **LLM answers**

  * IDs have the form:
    `<model>_q<question_id>_v<version>`
  * `model` is either:

    * `nlm` (NotebookLM)
    * `gemini` (Gemini)
  * For LLMs there are **two answer versions** per question (`v1`, `v2`).

* **Question IDs**

  * Stored with a trailing `.0` (e.g., `q1.0`) due to float handling in the original pipeline.
  * Conceptually, these are integers (`q1`, `q2`, …); the decimal part has no meaning and can be safely ignored or stripped.

This file also includes the judge’s decision for each pair, plus metadata used in the notebooks to compute win rates, BT scores, harmfulness, etc.

### `llm_as_judge_results_no_text.csv`

Results from the **LLM-as-judge** evaluation, where an LLM (e.g., *Llama-4 Maverick*) compares answers written by instructors and LLMs.

* Contains pairwise comparisons with:

  * Answer keys (consistent with the model ID conventions above),
  * Pair identifiers,
  * The LLM’s decision for each pair.
* Used primarily to fit Bradley–Terry models and generate LLM-as-judge ranking figures.

---

## Notebooks

Each notebook is designed to be **single-purpose** and to reproduce one or more figures from the paper. All notebooks assume the data files in `data/` are present and that they are run from the repository root.

### Core figures

* **`figure2_clean_full.ipynb` — Figure 2: Win rates vs models**

  * Generates the figure:

    > *Win rates of instructors compared to LLMs (Gemini and NotebookLM). Circles indicate win rates with 95% Wilson score confidence intervals; diamonds and squares represent win rates against NotebookLM and Gemini, respectively. Triangles indicate LLM-preference rates at the judge level. Models are shown at the top, followed by individual instructors.*
  * Uses pairwise comparison data to compute win rates, Wilson CIs, and judge-level LLM preferences, and plots them as a Cleveland dot plot.

* **`figure3_observed_vs_predicted_clean.ipynb` — Figure 3: Calibration of predicted vs observed win rates**

  * Generates:

    > *Calibration of model-predicted win probabilities vs. observed win rates (LLM vs. Instructors)…*
  * Uses predicted probabilities from the Δ-feature logit model (`pA_hat_logit`), focuses on **Gemini vs Instructor** and **NotebookLM vs Instructor** pairs, bins focal-model probabilities into deciles, computes observed win rates and Wilson CIs, and plots observed vs predicted with a $y=x$ calibration line.

* **`figure4_llm_as_judge_clean.ipynb` — Figure 4 / bt_judge: Bradley–Terry ranking with LLM as judge**

  * Generates:

    > *Bradley–Terry ranking of models with 95% confidence intervals, model strength summing to zero. Judge: Llama-4 Maverick. Each point shows the estimated latent strength; horizontal bars show maximum likelihood estimation confidence intervals from the observed Fisher information.*
  * Builds winner–loser comparisons from `llm_as_judge_results_no_text.csv`, fits a BT model with a sum-to-zero constraint, computes standard errors and CIs from the observed Fisher information, and plots a horizontal BT ranking.

### Appendix figures

* **`bt_ranking_D1.ipynb` — Figure D.1: BT strengths for Gemini, NotebookLM, and instructor average**

  * Generates:

    > *Bradley–Terry (BT) log-odds strength estimates for Gemini, NotebookLM, and the instructor average…*
  * Fits a BT model on main pairwise comparisons, extracts centered log-odds strengths, computes multiplicity-corrected simultaneous intervals from the χ² confidence ellipsoid, and plots the resulting coefficients and 95% intervals.

* **`win_rates_category_E1a_clean.ipynb` — Figure E.1a: Win rates by category**

  * Extends the main win-rate analysis by grouping comparisons by **question category** and plotting category-level win rates for LLMs vs instructors with appropriate confidence intervals.

* **`harmfulness_rate_F1_clean.ipynb` — Figure F.1: Harmfulness rates**

  * Generates:

    > *Harmfulness rates for instructors and LLMs. Points denote mean harmfulness rates … with non-parametric 95% bootstrap confidence intervals obtained by resampling responses.*
  * Applies the harmfulness definition from the Supplementary Information F, computes per-model harmfulness rates and bootstrap CIs, and plots them with point estimates and 95% intervals.
 
* **`strength_lojo_H1H2_clean.ipynb` — Figures H.1 & H.2: Judge strength and LOJO sensitivity**

  * **Figure H.1**

    > *Cross-judge trend between judges’ own strength as an instructor and calibration to human answer quality in LLM vs human matchups…*
    > Plots judge-level global win rate (own instructor strength) vs within-judge Spearman ρ between LOJO human-writer strength and LLM preference, with an OLS line and 95% SE band.

  * **Figure H.2**

    > *Judge level sensitivity to human instructor vs. LLM answers…*
    > Constructs a heatmap of the probability that a judge prefers the instructor over an LLM as a function of:

    * Judge’s own strength as a writer (win rate; horizontal axis),
    * Human answer’s LOJO win rate within the same pair (vertical axis).
      Uses empirical values on exact cells and a smoothed surface (Empirical-Bayes, inverse-distance interpolation) elsewhere.

* **`features_logit_figure_I1_clean.ipynb` — Figure I.1: Δ-feature logistic regression**

  * Generates:

    > *Estimated coefficients from a logistic regression of pairwise answer choices on differences in textual and pedagogical features, with two-way clustered standard errors (by judge and answer pair).*
  * Builds textual/pedagogical features (legal anchors, reasoning nuance, structure, tone, clarity, length, pedagogical support), constructs Δ-features (A–B), fits a logit with two-way clustered SEs, and plots coefficient estimates with 95% CIs.

* **`lojo_agreement_llm_judge_J1.ipynb` — Figure J.1: LOJO agreement with human consensus**

  * Generates:

    > *Leave-one-out agreement with the human consensus for each instructor and the LLM-as-a-judge…*
  * Computes, for each instructor and the LLM-as-judge:

    * Agreement rate with a leave-one-out majority of human judges on the same pairs.
    * Wilson 95% CIs.
  * Plots instructor agreement (blue circles) vs LLM agreement (orange squares), with instructors anonymized as “Instructor 1”–“Instructor 16” based on their main-experiment ranking.

---

## Requirements and usage

All notebooks are written in Python and run in Jupyter.

### Suggested environment

* Python ≥ 3.9
* Required packages (non-exhaustive):

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn` (for some plots)
  * `statsmodels`
  * `scipy`

You can install common dependencies with:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
```

### How to reproduce figures

1. Clone or download this repository.
2. Ensure the `data/` directory contains:

   * `pairwise_tutors_data_anonymized.csv`
   * `llm_as_judge_results_no_text.csv`
3. Open the desired notebook in `notebooks/` (e.g., `figure2_clean_full.ipynb`).
4. Run all cells from top to bottom.

   * Each notebook saves its figure into `figures/` (e.g., `figures/win_rates.png`, `figures/observed_vs_predicted.png`, etc.), which can then be included in the paper.

---

## Contact

For questions about the code or data, please contact:

* **Alejandro Salinas de León** — (contact details as in the manuscript)

```
```
