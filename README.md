# L#Square Mealy Implementation

This repository contains the implementation of the **L#Square** algorithm, developed for the Bachelor's Thesis: _"T.B.D."_ by Wouter Joosen (Radboud University, June 2026).

This codebase provides the implementation for learning (incomplete) Mealy Machines using L#Square.

## Prerequisites

Ensure you have your environment set up and the necessary dependencies installed:

    pip install -r requirements.txt

## Usage Guide

The project uses `lss.py` as the central entry point. You can run individual models, execute benchmark suites, or generate plots.

### 1. Run a Single Model

To learn a specific .dot Mealy Machine:

    python3 lss.py run path/to/model.dot

### 2. Run Benchmarks

You can execute the automated benchmark suites described in the thesis using the `benchmark` command. These map directly to the evaluation sections of the thesis:

| Command             | Thesis Reference | Description                                            |
| :------------------ | :--------------- | :----------------------------------------------------- |
| `protocols`         | Section 4.2.1    | Evaluates performance on standard protocol benchmarks. |
| `case-studies`      | Section 5.2      | Executes specific case study evaluations.              |
| `random-comparison` | Section 4.2.2    | Scaling experiments: L#Square vs. standard L#.         |
| `random-unknowns`   | Section 5.3      | Evaluates performance with an incomplete teacher.      |

**Example:** # Run the scaling benchmarks (as referenced in Section 4.2.2)

    python3 lss.py benchmark random-comparison

### 3. Generate Visualizations

After running the benchmarks, generate the plots used in the thesis:

    python3 lss.py plot

## Thesis Mapping

The benchmarks implemented in this repository map directly to the evaluation chapters of the thesis:

- **Standard Evaluation (Chapter 4):**
  - **Protocol Benchmarks:** Implemented via `benchmark protocols` (See Section 4.2.1).
  - **Random Scaling:** Implemented via `benchmark random-comparison` (See Section 4.2.2).

- **Evaluation on Incomplete Teacher (Chapter 5):**
  - **Case Studies:** Implemented via `benchmark case-studies` (See Section 5.2).
  - **Random Scaling:** Implemented via `benchmark random-unknowns` (See Section 5.3).
