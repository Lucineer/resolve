# Resolve

> Transmute intention into computation through multi-agent deliberation.

**58/58 tests passing** | Zero dependencies | Python 3.7+

## What It Does

Resolve takes human intention (plain text) and produces a running Python function through a multi-agent deliberation process. Not code generation — **deliberative compilation**.

```
resolve("sort a list of numbers in descending order")
  → 5 agents deliberate
  → confidence propagates via Bayesian fusion
  → converge when aggregate confidence > 0.85
  → artifact: working sort function
```

## Architecture

- **Payload**: JSON-first primitive — every operation is a `{op, inputs, constraints, confidence, provenance, metadata}` object
- **5 Agent Types**: IntentParser, Architect, Validator, Optimizer, MetaDeliberator
- **Consider/Resolve Protocol**: Agents propose, validate, optimize, and converge on solutions
- **Confidence Propagation**: Bayesian combination `1/(1/c1 + 1/c2)` — confidence flows through every operation
- **Forfeit Protocol**: Low-confidence agents transfer confidence to winners

## Quick Start

```python
from resolve import resolve

# Get a working function from intention
artifact = resolve("sort a list of numbers in descending order", verbose=True)
fn = artifact.to_executable()
result = fn([3, 1, 4, 1, 5, 9, 2, 6])
# → [9, 6, 5, 4, 3, 2, 1, 1]
```

## The Paradigm

```
Assembly → C → Java → JavaScript → Resolve
```

Where previous paradigms shifted *what we express*, Resolve shifts *why we express* — from implementation details to desired outcomes.

## Agents

| Agent | Role | Confidence Threshold |
|-------|------|---------------------|
| IntentParser | Extracts structured constraints from text | 0.9 |
| Architect | Proposes implementation approaches | 0.65 |
| Validator | Checks proposals against hard constraints | 0.8 |
| Optimizer | Suggests efficiency improvements | 0.6 |
| MetaDeliberator | Manages flow, detects convergence | 0.85 |

## File Structure

- `src/resolve.py` — Complete system (Payload, Agents, Engine, Artifact, CLI)
- `tests/test_resolve.py` — 58 tests

## From the A2A Future RA

This system implements concepts from the 5-round reverse-actualization at [github.com/Lucineer/a2a-future](https://github.com/Lucineer/a2a-future):
- Intention layer → structured constraints
- Deliberation layer → multi-agent proposal/consider/resolve
- Confidence propagation → Bayesian fusion
- Forfeit protocol → confidence transfer
- Artifact layer → executable output with provenance

Part of the [Lucineer ecosystem](https://github.com/Lucineer).
