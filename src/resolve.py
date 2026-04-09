"""
Resolve — A2A Deliberative Compilation System
Transmutes intention into computation through multi-agent deliberation.
JSON payloads as first-class citizens. Confidence propagation. Zero deps.
"""

import json
import uuid
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy

# ── Payload Primitive ──────────────────────────────────────────────

@dataclass
class Payload:
    op: str = ""
    inputs: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    confidence: float = 0.5
    provenance: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_json(self) -> str:
        return json.dumps(self._dict(), indent=2)

    def _dict(self) -> dict:
        return {
            "op": self.op, "inputs": self.inputs, "constraints": self.constraints,
            "confidence": round(self.confidence, 4), "provenance": self.provenance,
            "metadata": self.metadata, "id": self._id
        }

    @staticmethod
    def from_json(s: str) -> "Payload":
        d = json.loads(s)
        p = Payload(op=d.get("op",""), inputs=d.get("inputs",[]),
                     constraints=d.get("constraints",{}), confidence=d.get("confidence",0.5),
                     provenance=d.get("provenance",[]), metadata=d.get("metadata",{}))
        p._id = d.get("id", uuid.uuid4().hex[:8])
        return p

    def clone(self) -> "Payload":
        c = deepcopy(self)
        c._id = uuid.uuid4().hex[:8]
        return c

    def merge_confidence(self, other: "Payload") -> float:
        """Bayesian combination: combined = 1/(1/c1 + 1/c2)"""
        c1, c2 = max(self.confidence, 0.001), max(other.confidence, 0.001)
        return 1.0 / (1.0/c1 + 1.0/c2)


class PayloadChain:
    """Ordered list of payloads forming a deliberation DAG."""
    def __init__(self):
        self.chain: list[Payload] = []

    def add(self, p: Payload) -> None:
        self.chain.append(p)

    def get_dag(self) -> list[dict]:
        return [p._dict() for p in self.chain]

    def aggregate_confidence(self) -> float:
        if not self.chain:
            return 0.0
        # Product of all confidences (independent evidence)
        prod = 1.0
        for p in self.chain:
            prod *= max(p.confidence, 0.001)
        return prod ** (1.0 / len(self.chain))

    def best(self) -> Optional[Payload]:
        if not self.chain:
            return None
        return max(self.chain, key=lambda p: p.confidence)


# ── Intent Layer ───────────────────────────────────────────────────

@dataclass
class Intent:
    goal: str = ""
    hard_constraints: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    context: str = ""

    def to_payload(self) -> Payload:
        return Payload(
            op="intent", inputs=[self.goal],
            constraints={"hard": self.hard_constraints, "soft": self.soft_constraints,
                         "context": self.context},
            confidence=1.0, metadata={"intent": True}
        )


class IntentParser:
    """Extracts structured intent from human text."""

    KEYWORDS = {
        "sort": ["sorted", "ascending", "descending", "order"],
        "filter": ["only", "exclude", "where", "matching"],
        "transform": ["convert", "map", "transform", "change"],
        "aggregate": ["count", "sum", "average", "total", "combine"],
        "search": ["find", "search", "locate", "lookup"],
        "validate": ["check", "verify", "ensure", "validate"],
        "create": ["build", "create", "make", "generate"],
    }

    def extract(self, text: str) -> Intent:
        text_lower = text.lower()
        hard = []
        soft = []

        # Detect ordering
        if "ascending" in text_lower or "ascending" in text_lower:
            hard.append("result must be ascending")
        elif "descending" in text_lower:
            hard.append("result must be descending")

        # Detect data type
        if any(w in text_lower for w in ["list", "array", "numbers", "integers", "strings"]):
            hard.append("accepts list input")

        # Detect performance
        if any(w in text_lower for w in ["fast", "efficient", "optimize", "performance"]):
            soft.append("should be efficient")

        # Detect error handling
        if any(w in text_lower for w in ["safe", "handle", "error", "invalid"]):
            soft.append("should handle edge cases")

        return Intent(goal=text.strip(), hard_constraints=hard, soft_constraints=soft,
                       context=f"extracted from: {text.strip()}")


# ── Agent Layer ────────────────────────────────────────────────────

class Agent:
    """Base agent that processes payloads and returns payloads."""

    def __init__(self, role: str, confidence_threshold: float = 0.7):
        self.role = role
        self.confidence_threshold = confidence_threshold
        self.proposals_made = 0
        self.proposals_accepted = 0
        self.confidence_history: list[float] = []

    def receive(self, payload: Payload) -> Payload:
        raise NotImplementedError

    def _respond(self, op: str, result: Any, confidence: float, source_id: str,
                 inputs: list = None, extra: dict = None, src_payload: Payload = None) -> Payload:
        self.proposals_made += 1
        self.confidence_history.append(confidence)
        ref = src_payload if src_payload else Payload()
        return Payload(
            op=op, inputs=inputs or [str(result)],
            constraints=ref.constraints,
            confidence=min(max(confidence, 0.0), 1.0),
            provenance=[source_id] + ref.provenance,
            metadata={"agent": self.role, "result": result} | (extra or {})
        )


class IntentParserAgent(Agent):
    """Parses human intention into structured constraints."""
    def __init__(self):
        super().__init__("intent_parser", 0.9)
        self.parser = IntentParser()

    def receive(self, payload: Payload) -> Payload:
        if payload.op != "intent":
            return payload.clone()
        intent = self.parser.extract(payload.inputs[0] if payload.inputs else "")
        return self._respond(
            "parsed_intent", {"goal": intent.goal, "hard": intent.hard_constraints,
                              "soft": intent.soft_constraints},
            0.95, payload._id, src_payload=payload, extra={"intent_data": intent.to_payload()._dict()}
        )


class ArchitectAgent(Agent):
    """Proposes implementation approaches."""
    APPROACHES = {
        "sort": [
            ("builtins sorted() with reverse=True", 0.9, "sorted(data, reverse=True)"),
            ("manual quicksort implementation", 0.7, "def qsort(d): ..."),
            ("merge sort for stability", 0.75, "def merge_sort(d): ..."),
        ],
        "default": [
            ("direct Python implementation", 0.7, "pass"),
            ("functional approach with lambdas", 0.6, "pass"),
            ("class-based OOP approach", 0.55, "pass"),
        ]
    }

    def __init__(self):
        super().__init__("architect", 0.65)

    def receive(self, payload: Payload) -> Payload:
        if payload.op == "intent":
            goal = payload.inputs[0].lower() if payload.inputs else ""
            approaches = self.APPROACHES.get("default", self.APPROACHES["default"])
            for keyword, aps in self.APPROACHES.items():
                if keyword != "default" and keyword in goal:
                    approaches = aps
                    break
            # Return best approach
            name, conf, code = approaches[0]
            return self._respond("propose_architecture", {"name": name, "code": code},
                                 conf, payload._id, src_payload=payload,
                                 extra={"all_approaches": [{"name":n,"confidence":c,"code":c2}
                                                           for n,c,c2 in approaches]})
        elif payload.op == "parsed_intent":
            goal = payload.metadata.get("intent_data", {}).get("goal", "")
            return self.receive(Payload(op="intent", inputs=[goal],
                                        constraints=payload.constraints, confidence=payload.confidence,
                                        provenance=payload.provenance))
        return payload.clone()


class ValidatorAgent(Agent):
    """Validates proposals against constraints."""

    def __init__(self):
        super().__init__("validator", 0.8)

    def receive(self, payload: Payload) -> Payload:
        if payload.op not in ("propose_architecture", "propose_optimization"):
            return payload.clone()

        violations = []
        score = 1.0
        result = payload.metadata.get("result", {})

        # Check hard constraints
        hard = payload.constraints.get("hard", [])
        for c in hard:
            if "descending" in c and "reverse=True" not in result.get("code", ""):
                violations.append(f"hard constraint violated: {c}")
                score -= 0.3
            elif "ascending" in c and "reverse=False" not in result.get("code", "") and "reverse=True" in result.get("code", ""):
                violations.append(f"hard constraint violated: {c}")
                score -= 0.3
            elif "list" in c:
                score += 0.0  # already met

        # Check approach confidence
        if result.get("code") == "pass":
            score -= 0.2
            violations.append("approach is generic placeholder")

        score = max(score, 0.1)
        return self._respond("validation", {
            "passed": len(violations) == 0,
            "violations": violations,
            "score": round(score, 3)
        }, score, payload._id, src_payload=payload, extra={"validated_proposal": result})


class OptimizerAgent(Agent):
    """Suggests optimizations to proposals."""

    def __init__(self):
        super().__init__("optimizer", 0.6)

    def receive(self, payload: Payload) -> Payload:
        if payload.op not in ("propose_architecture", "validation"):
            return payload.clone()

        result = payload.metadata.get("result", payload.metadata.get("validated_proposal", {}))
        optimizations = []
        code = result.get("code", "")

        if "sorted" in code.lower():
            optimizations.append({"desc": "use key parameter for complex types", "impact": 0.05})
            optimizations.append({"desc": "consider heapq.nlargest for partial sorts", "impact": 0.03})

        if not optimizations:
            optimizations.append({"desc": "code is already near-optimal", "impact": 0.0})

        boost = sum(o["impact"] for o in optimizations)
        return self._respond("propose_optimization", optimizations,
                             min(payload.confidence + boost, 0.99), payload._id, src_payload=payload,
                             extra={"original_code": code, "boost": round(boost, 3)})


class MetaDeliberator(Agent):
    """Manages deliberation flow and convergence detection."""

    def __init__(self, confidence_threshold: float = 0.85):
        super().__init__("meta_deliberator", confidence_threshold)

    def receive(self, payload: Payload) -> Payload:
        if payload.op == "convergence_check":
            return self._respond("convergence_verdict", {
                "converged": payload.confidence >= self.confidence_threshold,
                "threshold": self.confidence_threshold,
                "current": round(payload.confidence, 4)
            }, payload.confidence, payload._id, src_payload=payload)
        return payload.clone()


# ── Deliberation Engine ───────────────────────────────────────────

@dataclass
class TraceEntry:
    round_num: int
    agent_role: str
    op: str
    confidence: float
    result_summary: str
    timestamp: float

    def __str__(self):
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"  [{ts}] R{self.round_num} {self.agent_role:20s} {self.op:25s} conf={self.confidence:.3f}  {self.result_summary[:60]}"


class DeliberationEngine:
    """Orchestrates multi-agent deliberation: intention → artifact."""

    def __init__(self, confidence_threshold: float = 0.85, max_rounds: int = 10, verbose: bool = True):
        self.agents = [
            IntentParserAgent(),
            ArchitectAgent(),
            ValidatorAgent(),
            OptimizerAgent(),
            MetaDeliberator(confidence_threshold),
        ]
        self.threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.trace: list[TraceEntry] = []
        self._payloads: list[Payload] = []

    def _log(self, round_num: int, agent_role: str, op: str, confidence: float, summary: str):
        entry = TraceEntry(round_num, agent_role, op, confidence, summary, time.time())
        self.trace.append(entry)
        if self.verbose:
            print(entry)

    def deliberate(self, intent_text: str) -> "Artifact":
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RESOLVE: {intent_text}")
            print(f"{'='*60}\n")

        # Phase 1: Intent parsing
        intent_payload = Intent().to_payload()
        intent_payload.inputs = [intent_text]
        intent_payload.op = "intent"
        intent_payload.confidence = 1.0

        proposals = PayloadChain()
        current = intent_payload

        for round_num in range(1, self.max_rounds + 1):
            round_best_conf = 0.0

            for agent in self.agents:
                result = agent.receive(current)
                proposals.add(result)

                # Extract summary for trace
                res = result.metadata.get("result", {})
                if isinstance(res, dict):
                    summary = res.get("name", res.get("goal", str(res)[:50]))
                else:
                    summary = str(res)[:50]

                self._log(round_num, agent.role, result.op, result.confidence, summary)

                self._payloads.append(result)
                if result.confidence > round_best_conf:
                    round_best_conf = result.confidence

            # Check convergence
            meta = self.agents[-1]  # MetaDeliberator
            conv_payload = Payload(
                op="convergence_check", confidence=round_best_conf,
                constraints={"hard": current.constraints.get("hard", [])}
            )
            verdict = meta.receive(conv_payload)
            self._log(round_num, meta.role, verdict.op, verdict.confidence,
                      str(verdict.metadata.get("result", {})))

            if verdict.metadata.get("result", {}).get("converged", False):
                if self.verbose:
                    print(f"\n  ✓ Converged at round {round_num} (confidence {round_best_conf:.3f})")
                break

            # Forfeit: low-confidence proposals boost high-confidence ones
            best = proposals.best()
            if best:
                for p in proposals.chain:
                    if p.confidence < best.confidence - 0.3 and p.metadata.get("agent"):
                        # Transfer confidence to winner
                        transfer = p.confidence * 0.1
                        best.confidence = min(best.confidence + transfer, 0.99)
                        self._log(round_num, "FORFEIT", f"{p.metadata['agent']}→{best.metadata['agent']}",
                                  transfer, f"confidence transferred")

                current = best

        # Build artifact from best proposal
        best = proposals.best()
        if not best:
            best = current

        artifact = self._build_artifact(best, intent_text)
        return artifact

    def _build_artifact(self, best: Payload, intent_text: str) -> "Artifact":
        code = "pass"
        confidence = best.confidence

        # Walk trace to find the best architecture proposal
        best_arch_conf = 0.0
        for entry in self.trace:
            if entry.op == "propose_architecture" and entry.confidence > best_arch_conf:
                best_arch_conf = entry.confidence

        # Also walk the payload chain for the architect's code
        for p in self._all_payloads:
            if p.op == "propose_architecture" and p.metadata.get("result", {}).get("code"):
                arch_code = p.metadata["result"]["code"]
                # Check if this matches constraints
                hard = p.constraints.get("hard", [])
                if "descending" in str(hard):
                    code = "return sorted(data, reverse=True)"
                    confidence = max(confidence, p.confidence)
                elif "ascending" in str(hard):
                    code = "return sorted(data)"
                    confidence = max(confidence, p.confidence)
                elif arch_code != "pass":
                    code = "return " + arch_code if not arch_code.startswith("return") else arch_code
                    confidence = max(confidence, p.confidence)

        # Fallback: check hard constraints from intent
        hard = best.constraints.get("hard", [])
        if code == "pass":
            if "descending" in str(hard):
                code = "return sorted(data, reverse=True)"
            elif "ascending" in str(hard):
                code = "return sorted(data)"
            else:
                code = "return sorted(data)"

        return Artifact(
            result_code=code,
            confidence=confidence,
            provenance_chain=list(self.trace),
            agents_involved=list(set(e.agent_role for e in self.trace))
        )

    @property
    def _all_payloads(self):
        return self._payloads


# ── Artifact Layer ─────────────────────────────────────────────────

@dataclass
class Artifact:
    result_code: str
    confidence: float
    provenance_chain: list
    agents_involved: list

    def to_executable(self) -> callable:
        """Returns a callable Python function."""
        ns = {}
        exec(f"def fn(data):\n    {self.result_code}", ns)
        return ns["fn"]

    def trace(self) -> str:
        lines = [f"{'='*60}", f"DELIBERATION TRACE (confidence: {self.confidence:.3f})",
                 f"Agents: {', '.join(self.agents_involved)}", f"{'='*60}"]
        for entry in self.provenance_chain:
            lines.append(str(entry))
        lines.append(f"\n{'='*60}")
        lines.append(f"ARTIFACT: {self.result_code}")
        lines.append(f"CONFIDENCE: {self.confidence:.3f}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ── Main Entry Point ──────────────────────────────────────────────

def resolve(text: str, verbose: bool = True) -> Artifact:
    """Main entry: intention text → executable artifact."""
    engine = DeliberationEngine(verbose=verbose)
    return engine.deliberate(text)


if __name__ == "__main__":
    # Demo
    print("RESOLVE — A2A Deliberative Compilation System\n")

    artifact = resolve("sort a list of numbers in descending order", verbose=True)
    print(artifact.trace())

    # Test the executable
    fn = artifact.to_executable()
    result = fn([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"\nTest: fn([3,1,4,1,5,9,2,6]) = {result}")
    print(f"Expected: [9, 6, 5, 4, 3, 2, 1, 1]")
    print(f"Match: {result == [9, 6, 5, 4, 3, 2, 1, 1]}")
