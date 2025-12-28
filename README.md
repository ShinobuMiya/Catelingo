# Catelingo

**Catelingo** is a constraint-based semantic validity verifier for language model outputs.  
It detects *semantic no-go cases*—sentences that are syntactically fluent but semantically impossible—by checking **explicit semantic constraints**, without relying on reasoning chains, factual knowledge bases, or model retraining.

This repository provides a **toy implementation** accompanying the paper:

> **Catelingo: Constraint-Based Semantic Validity Verification for Language Models**  
> Shinobu Miya, 2025  
> (arXiv preprint)

---

## Motivation

Large Language Models (LLMs) often generate outputs that are grammatically correct yet semantically invalid, such as:

- temporal impossibilities  
- numerical range violations  
- semantic type clashes  

These failures do **not** require reasoning chains or external knowledge to detect.  
They arise from violations of basic semantic constraints.

Catelingo addresses this class of errors by reframing **semantic validity as constraint satisfiability**.

---

## Core Idea

Catelingo treats semantic validity as a **constraint satisfaction problem (CSP)**:

- Lexical senses and grammatical relations induce **explicit semantic constraints**
- Constraints are propagated along syntactic dependency edges
- A sentence is:
  - **SAT** if at least one interpretation satisfies all constraints
  - **UNSAT** if all interpretations violate constraints
  - **UNKNOWN** if satisfiability cannot be determined due to underspecification

Importantly:

- No reasoning chains are generated
- No factual knowledge is stored or retrieved
- No single interpretation is selected or preferred
- Ambiguity is preserved unless all possibilities fail

Semantic validity is defined purely as **existence of a consistent interpretation**.

---

## What Catelingo Does *Not* Do

Catelingo deliberately does **not**:

- perform logical reasoning or inference
- check factual correctness
- retrieve external knowledge
- rank interpretations by plausibility
- repair or rewrite invalid outputs

It is a **verification layer**, not a generator or interpreter.

---

## Quick Start

Requires Python 3.8+. No heavy dependencies.

```bash
# Clone the repository
git clone https://github.com/ShinobuMiya/Catelingo.git
cd Catelingo

# Run the test suite (reproduces all paper examples)
python -m unittest discover tests
```

---

## Implementation Overview

This repository contains a **minimal toy implementation** designed to demonstrate feasibility and clarity.

Key properties:

- Pure Python implementation
- Small, sense-level lexicon
- Explicit semantic constraints (type, temporal, numerical)
- Dependency-local constraint propagation (AC-3–like)
- Deterministic behavior

### Pipeline

Input sentence
↓
Dependency parsing (upstream)
↓
Sense candidates + constraint instantiation
↓
Constraint propagation on dependency graph
↓
Verdict: SAT / UNSAT / UNKNOWN


---

## Examples

### Temporal Constraint Violation (UNSAT)

> “The Eiffel Tower was built in 1889 to commemorate World War I.”

Violates the semantic constraint:

construct.time ≥ commemorate.event.time


→ **UNSAT**

---

### Numerical Constraint Violation (UNSAT)

> “The probability of success is 150%.”

Violates:

```math
probability ∈ [0, 1]
```

→ **UNSAT**

---

### Semantic Type Clash (UNSAT)

> “The absolute monarchy of bananas is like a spaceship dancing.”

No allowed type unification under the active constraint profile.

→ **UNSAT**

This is a **compositional failure**:  
> no single word is invalid in isolation, but the combined constraints admit no coherent refinement.

---

### Controlled Metaphor (SAT)

> “Autumn makes your cheek like an apple.”

A single explicit relaxation rule is enabled:

```math
APPLE_fruit → RED_OBJECT
```

→ **SAT**

Metaphor is treated as a **structured semantic operator**, not an exception or loophole.

---

## Constraint Profiles

Semantic validity is **profile-dependent**, not absolute.

Catelingo supports domain-specific constraint profiles:

- English / general (implemented)
- English / finance (conceptual)
- English / poetry (conceptual)

Profiles determine:

- which constraints are enforced
- which type relaxations are permitted
- strictness of evaluation

Domain adaptation is achieved by **switching profiles**, not retraining models.

---

## Relationship to Other Verification Methods

Catelingo is **orthogonal and complementary** to:

- reasoning-based verification (Chain-of-Thought, Self-Consistency)
- knowledge-based verification (RAG, fact checking)
- structural verification (e.g. Eidoku)

It targets semantic failures that **do not manifest as reasoning or factual errors**.

---

## Reproducibility

- All examples in the paper are covered by automated tests
- Constraint propagation is fully deterministic
- A GitHub Actions CI pipeline validates expected outcomes

---

## Repository Structure

.
├── src/
│   ├── engine.py       # Constraint propagation logic (AC-3 like)
│   ├── models.py       # Data structures (Sense, Constraint, etc.)
│   └── loader.py       # YAML/JSON loader
├── scenarios/          # Input examples (JSON)
├── data/               # Lexicon and Constraint definitions
├── tests/              # Reproduction scripts
├── README.md
└── LICENSE


---

## Status

This is a **toy implementation** intended for:

- conceptual validation
- clarity of semantics
- reproducibility of examples

It is **not** intended as a production-ready semantic parser.

---

## Citation

If you use this work, please cite:

@article{miya2025catelingo,
title={Catelingo: Constraint-Based Semantic Validity Verification for Language Models},
author={Miya, Shinobu},
year={2025},
journal={arXiv preprint}
}


---

## License

Apache 2.0 License.
