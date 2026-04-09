#!/usr/bin/env python3
"""Resolve — Functional A2A Deliberative Compilation System

A resolve program (.rsn) turns human intention into working code through
multi-agent deliberation. Every value is a Payload (JSON-native with confidence).

Usage:
    resolve "sort a list of numbers descending"
    resolve "parse a CSV and find the average of column 3"
    resolve build program.rsn
    resolve run program.rsn
"""

import json, sys, hashlib, time, re, os
from typing import Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# ============================================================
# CORE TYPES — Payload (JSON-native with confidence)
# ============================================================

class Payload:
    """Every value in Resolve carries confidence. Payloads ARE JSON."""
    
    def __init__(self, data, confidence: float = 1.0, source: str = "", provenance: list = None):
        self.data = data
        self.confidence = max(0.0, min(1.0, confidence))
        self.source = source
        self.provenance = provenance or []
    
    def __repr__(self):
        return f"Payload({self.data!r}, conf={self.confidence:.3f}, src={self.source!r})"
    
    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "confidence": self.confidence,
            "source": self.source,
            "provenance": self.provenance,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Payload":
        return cls(d["data"], d.get("confidence", 1.0), d.get("source", ""), d.get("provenance", []))
    
    def with_source(self, source: str) -> "Payload":
        return Payload(self.data, self.confidence, source, self.provenance)
    
    def with_confidence(self, conf: float) -> "Payload":
        return Payload(self.data, conf, self.source, self.provenance)
    
    def is_number(self) -> bool:
        return isinstance(self.data, (int, float))
    
    def is_string(self) -> bool:
        return isinstance(self.data, str)
    
    def is_list(self) -> bool:
        return isinstance(self.data, list)
    
    def is_dict(self) -> bool:
        return isinstance(self.data, dict)
    
    def is_null(self) -> bool:
        return self.data is None
    
    def number(self) -> float:
        if self.is_number():
            return float(self.data)
        raise TypeError(f"Expected number, got {type(self.data).__name__}")
    
    def string(self) -> str:
        if self.is_string():
            return self.data
        raise TypeError(f"Expected string, got {type(self.data).__name__}")
    
    def list(self) -> list:
        if self.is_list():
            return self.data
        raise TypeError(f"Expected list, got {type(self.data).__name__}")


class PayloadType(Enum):
    NUMBER = "number"
    STRING = "string"
    LIST = "list"
    DICT = "dict"
    BOOL = "bool"
    NULL = "null"
    ANY = "any"
    FUNCTION = "function"


def infer_type(p: Payload) -> PayloadType:
    if p.is_number(): return PayloadType.NUMBER
    if p.is_string(): return PayloadType.STRING
    if p.is_list(): return PayloadType.LIST
    if p.is_dict(): return PayloadType.DICT
    if isinstance(p.data, bool): return PayloadType.BOOL
    if p.is_null(): return PayloadType.NULL
    return PayloadType.ANY


def type_check(payload: Payload, expected: PayloadType) -> tuple[bool, str]:
    actual = infer_type(payload)
    if expected == PayloadType.ANY:
        return (True, "")
    if expected == PayloadType.NUMBER and payload.is_number():
        return (True, "")
    if expected == PayloadType.STRING and payload.is_string():
        return (True, "")
    if expected == PayloadType.LIST and payload.is_list():
        return (True, "")
    if expected == PayloadType.DICT and payload.is_dict():
        return (True, "")
    if expected == PayloadType.BOOL and isinstance(payload.data, bool):
        return (True, "")
    if expected == PayloadType.NULL and payload.is_null():
        return (True, "")
    return (False, f"Expected {expected.value}, got {actual.value}")


# ============================================================
# BAYESIAN CONFIDENCE COMBINATION
# ============================================================

def bayesian_combine(c1: float, c2: float) -> float:
    """Combine two independent confidence sources: 1/(1/c1 + 1/c2)"""
    c1, c2 = max(c1, 0.001), max(c2, 0.001)
    return 1.0 / (1.0/c1 + 1.0/c2)

def bayesian_multi(confidences: list[float]) -> float:
    """Combine multiple confidence sources."""
    result = confidences[0] if confidences else 0.0
    for c in confidences[1:]:
        result = bayesian_combine(result, c)
    return result

def confidence_decay(conf: float, rounds: int, decay_rate: float = 0.95) -> float:
    """Decay confidence over rounds."""
    return conf * (decay_rate ** rounds)


# ============================================================
# AGENT TYPES — the deliberation participants
# ============================================================

class AgentRole(Enum):
    INTENT_PARSER = "intent_parser"
    ARCHITECT = "architect"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"
    CODEGEN = "codegen"
    META_DELIBERATOR = "meta_deliberator"


class AgentProposal:
    """A proposal from an agent in deliberation."""
    
    def __init__(self, agent: str, role: AgentRole, action: str, payload: Payload,
                 reasoning: str = "", alternatives: list = None):
        self.agent = agent
        self.role = role
        self.action = action
        self.payload = payload
        self.reasoning = reasoning
        self.alternatives = alternatives or []
        self.timestamp = time.time()
        self.considered_by: list[str] = []
        self.resolved = False
        self.accepted = False


class DeliberationRound:
    """One round of deliberation."""
    
    def __init__(self, round_num: int, intent: Payload):
        self.round_num = round_num
        self.intent = intent
        self.proposals: list[AgentProposal] = []
        self.considerations: list[dict] = []
        self.forfeits: list[dict] = []
        self.aggregate_confidence = 0.0
        self.converged = False
        self.output: Optional[Payload] = None


# ============================================================
# BUILT-IN OPERATIONS — the primitive computation layer
# ============================================================

class Operation:
    """A built-in operation in the Resolve system."""
    
    def __init__(self, name: str, inputs: list[PayloadType], output: PayloadType,
                 fn: callable, cost: float = 0.0, description: str = ""):
        self.name = name
        self.inputs = inputs
        self.output = output
        self.fn = fn
        self.cost = cost
        self.description = description
    
    def __call__(self, *args: Payload) -> Payload:
        result, confidence = self.fn(*args)
        if not isinstance(result, Payload):
            result = Payload(result, confidence)
        return result


OPERATIONS: dict[str, Operation] = {}

def register_op(name: str, inputs: list[PayloadType], output: PayloadType,
                fn: callable, cost: float = 0.0, desc: str = ""):
    OPERATIONS[name] = Operation(name, inputs, output, fn, cost, desc)

# --- Arithmetic ---
def _add(a: Payload, b: Payload):
    ok, err = type_check(a, PayloadType.NUMBER)
    if not ok: return Payload(None, 0.0, "type_error"), 0.0
    ok, err = type_check(b, PayloadType.NUMBER)
    if not ok: return Payload(None, 0.0, "type_error"), 0.0
    return Payload(a.number() + b.number(), bayesian_combine(a.confidence, b.confidence), "add"), bayesian_combine(a.confidence, b.confidence)

def _sub(a: Payload, b: Payload):
    return Payload(a.number() - b.number(), bayesian_combine(a.confidence, b.confidence), "sub"), bayesian_combine(a.confidence, b.confidence)

def _mul(a: Payload, b: Payload):
    return Payload(a.number() * b.number(), bayesian_combine(a.confidence, b.confidence), "mul"), bayesian_combine(a.confidence, b.confidence)

def _div(a: Payload, b: Payload):
    if abs(b.number()) < 1e-10:
        return Payload(None, 0.0, "division_by_zero"), 0.0
    return Payload(a.number() / b.number(), bayesian_combine(a.confidence, b.confidence) * 0.98, "div"), bayesian_combine(a.confidence, b.confidence) * 0.98

def _pow(a: Payload, b: Payload):
    try:
        result = a.number() ** b.number()
        conf = bayesian_combine(a.confidence, b.confidence) * 0.97
        return Payload(result, conf, "pow"), conf
    except (OverflowError, ValueError):
        return Payload(None, 0.0, "overflow"), 0.0

def _mod(a: Payload, b: Payload):
    if abs(b.number()) < 1e-10:
        return Payload(None, 0.0, "mod_by_zero"), 0.0
    return Payload(a.number() % b.number(), bayesian_combine(a.confidence, b.confidence), "mod"), bayesian_combine(a.confidence, b.confidence)

# --- Comparison ---
def _eq(a: Payload, b: Payload):
    result = a.data == b.data
    conf = bayesian_combine(a.confidence, b.confidence)
    return Payload(result, conf, "eq"), conf

def _neq(a: Payload, b: Payload):
    return _eq(a, b)[0].data == (a.data != b.data), bayesian_combine(a.confidence, b.confidence)

def _gt(a: Payload, b: Payload):
    return Payload(a.number() > b.number(), bayesian_combine(a.confidence, b.confidence), "gt"), bayesian_combine(a.confidence, b.confidence)

def _gte(a: Payload, b: Payload):
    return Payload(a.number() >= b.number(), bayesian_combine(a.confidence, b.confidence), "gte"), bayesian_combine(a.confidence, b.confidence)

def _lt(a: Payload, b: Payload):
    return Payload(a.number() < b.number(), bayesian_combine(a.confidence, b.confidence), "lt"), bayesian_combine(a.confidence, b.confidence)

def _lte(a: Payload, b: Payload):
    return Payload(a.number() <= b.number(), bayesian_combine(a.confidence, b.confidence), "lte"), bayesian_combine(a.confidence, b.confidence)

# --- List operations ---
def _sort_asc(lst: Payload):
    items = lst.list()
    if not items: return Payload([], lst.confidence, "sort_asc"), lst.confidence
    if all(isinstance(x, (int, float)) for x in items):
        return Payload(sorted(items), lst.confidence * 0.99, "sort_asc"), lst.confidence * 0.99
    if all(isinstance(x, str) for x in items):
        return Payload(sorted(items), lst.confidence * 0.99, "sort_asc"), lst.confidence * 0.99
    return Payload(None, 0.0, "mixed_types"), 0.0

def _sort_desc(lst: Payload):
    result, conf = _sort_asc(lst)
    if result.data is not None:
        result.data = list(reversed(result.data))
        result.source = "sort_desc"
    return result, conf

def _reverse(lst: Payload):
    return Payload(list(reversed(lst.list())), lst.confidence, "reverse"), lst.confidence

def _first(lst: Payload):
    items = lst.list()
    if items: return Payload(items[0], lst.confidence, "first"), lst.confidence
    return Payload(None, 0.0, "empty_list"), 0.0

def _last(lst: Payload):
    items = lst.list()
    if items: return Payload(items[-1], lst.confidence, "last"), lst.confidence
    return Payload(None, 0.0, "empty_list"), 0.0

def _rest(lst: Payload):
    items = lst.list()
    if items: return Payload(items[1:], lst.confidence, "rest"), lst.confidence
    return Payload([], lst.confidence, "rest"), lst.confidence

def _count(lst: Payload):
    return Payload(len(lst.list()), lst.confidence, "count"), lst.confidence

def _filter(lst: Payload, fn_payload: Payload):
    """Filter with a predicate payload containing 'condition' key."""
    items = lst.list()
    cond = fn_payload.data if isinstance(fn_payload.data, dict) else {}
    threshold = cond.get("gt")
    lt = cond.get("lt")
    eq_val = cond.get("eq")
    contains = cond.get("contains")
    field = cond.get("field")
    
    result = []
    for item in items:
        val = item.get(field) if (field and isinstance(item, dict)) else item
        if threshold is not None and isinstance(val, (int, float)) and val > threshold:
            result.append(item)
        elif lt is not None and isinstance(val, (int, float)) and val < lt:
            result.append(item)
        elif eq_val is not None and val == eq_val:
            result.append(item)
        elif contains is not None and isinstance(val, str) and contains in val:
            result.append(item)
    
    conf = lst.confidence * 0.95
    return Payload(result, conf, "filter"), conf

def _map(lst: Payload, fn_payload: Payload):
    """Map with an operation name from payload."""
    items = lst.list()
    op_name = fn_payload.data if isinstance(fn_payload.data, str) else fn_payload.string()
    if op_name in OPERATIONS:
        result = [OPERATIONS[op_name].fn(Payload(x, lst.confidence))[0].data for x in items]
        return Payload(result, lst.confidence * 0.95, "map"), lst.confidence * 0.95
    return Payload(None, 0.0, f"unknown_op:{op_name}"), 0.0

def _reduce(lst: Payload, fn_payload: Payload):
    """Reduce with accumulator."""
    items = lst.list()
    op_name = fn_payload.data if isinstance(fn_payload.data, str) else fn_payload.string()
    if op_name in OPERATIONS and items:
        acc = items[0]
        for item in items[1:]:
            p, _ = OPERATIONS[op_name].fn(Payload(acc), Payload(item))
            if p.data is None: return Payload(None, 0.0, "reduce_error"), 0.0
            acc = p.data
        return Payload(acc, lst.confidence * 0.95, "reduce"), lst.confidence * 0.95
    return Payload(None, 0.0, "reduce_noop"), 0.0

def _flatten(lst: Payload):
    result = []
    for item in lst.list():
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return Payload(result, lst.confidence * 0.98, "flatten"), lst.confidence * 0.98

def _unique(lst: Payload):
    seen = set()
    result = []
    for item in lst.list():
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return Payload(result, lst.confidence * 0.99, "unique"), lst.confidence * 0.99

def _slice(lst: Payload, start: Payload, end: Payload):
    items = lst.list()
    s = int(start.number()) if start.is_number() else 0
    e = int(end.number()) if end.is_number() else len(items)
    return Payload(items[s:e], lst.confidence, "slice"), lst.confidence

def _zip(a: Payload, b: Payload):
    result = list(zip(a.list(), b.list()))
    conf = bayesian_combine(a.confidence, b.confidence)
    return Payload(result, conf, "zip"), conf

# --- String operations ---
def _upper(s: Payload):
    return Payload(s.string().upper(), s.confidence, "upper"), s.confidence

def _lower(s: Payload):
    return Payload(s.string().lower(), s.confidence, "lower"), s.confidence

def _strip(s: Payload):
    return Payload(s.string().strip(), s.confidence, "strip"), s.confidence

def _split(s: Payload, sep: Payload):
    return Payload(s.string().split(sep.string()), bayesian_combine(s.confidence, sep.confidence), "split"), bayesian_combine(s.confidence, sep.confidence)

def _join(lst: Payload, sep: Payload):
    items = [str(x) for x in lst.list()]
    return Payload(sep.string().join(items), bayesian_combine(lst.confidence, sep.confidence), "join"), bayesian_combine(lst.confidence, sep.confidence)

def _replace(s: Payload, old: Payload, new: Payload):
    return Payload(s.string().replace(old.string(), new.string()), bayesian_combine(s.confidence, bayesian_combine(old.confidence, new.confidence)), "replace"), bayesian_combine(s.confidence, bayesian_combine(old.confidence, new.confidence))

def _len(s: Payload):
    if s.is_list(): return Payload(len(s.list()), s.confidence, "len"), s.confidence
    if s.is_string(): return Payload(len(s.string()), s.confidence, "len"), s.confidence
    if s.is_dict(): return Payload(len(s.data), s.confidence, "len"), s.confidence
    return Payload(None, 0.0, "type_error"), 0.0

def _contains(haystack: Payload, needle: Payload):
    if haystack.is_string() and needle.is_string():
        return Payload(needle.string() in haystack.string(), bayesian_combine(haystack.confidence, needle.confidence), "contains"), bayesian_combine(haystack.confidence, needle.confidence)
    if haystack.is_list():
        return Payload(needle.data in haystack.list(), bayesian_combine(haystack.confidence, needle.confidence), "contains"), bayesian_combine(haystack.confidence, needle.confidence)
    return Payload(False, 0.5, "contains"), 0.5

def _starts_with(s: Payload, prefix: Payload):
    return Payload(s.string().startswith(prefix.string()), bayesian_combine(s.confidence, prefix.confidence), "starts_with"), bayesian_combine(s.confidence, prefix.confidence)

def _ends_with(s: Payload, suffix: Payload):
    return Payload(s.string().endswith(suffix.string()), bayesian_combine(s.confidence, suffix.confidence), "ends_with"), bayesian_combine(s.confidence, suffix.confidence)

# --- Dict operations ---
def _get(d: Payload, key: Payload):
    if d.is_dict():
        val = d.data.get(key.data if not key.is_string() else key.string())
        if val is not None:
            return Payload(val, d.confidence * 0.99, "get"), d.confidence * 0.99
    return Payload(None, 0.0, "key_not_found"), 0.0

def _keys(d: Payload):
    if d.is_dict():
        return Payload(list(d.data.keys()), d.confidence, "keys"), d.confidence
    return Payload(None, 0.0, "not_dict"), 0.0

def _values(d: Payload):
    if d.is_dict():
        return Payload(list(d.data.values()), d.confidence, "values"), d.confidence
    return Payload(None, 0.0, "not_dict"), 0.0

def _merge(a: Payload, b: Payload):
    if a.is_dict() and b.is_dict():
        merged = {**a.data, **b.data}
        return Payload(merged, bayesian_combine(a.confidence, b.confidence), "merge"), bayesian_combine(a.confidence, b.confidence)
    return Payload(None, 0.0, "merge_not_dicts"), 0.0

# --- Aggregation ---
def _sum(lst: Payload):
    if not lst.is_list(): return Payload(None, 0.0, "not_list"), 0.0
    nums = [x for x in lst.list() if isinstance(x, (int, float))]
    return Payload(sum(nums), lst.confidence * (len(nums)/max(len(lst.list()),1)), "sum"), lst.confidence * (len(nums)/max(len(lst.list()),1))

def _mean(lst: Payload):
    if not lst.is_list(): return Payload(None, 0.0, "not_list"), 0.0
    nums = [x for x in lst.list() if isinstance(x, (int, float))]
    if not nums: return Payload(None, 0.0, "no_numbers"), 0.0
    return Payload(sum(nums)/len(nums), lst.confidence * (len(nums)/max(len(lst.list()),1)) * 0.99, "mean"), lst.confidence * 0.99

def _min(lst: Payload):
    nums = [x for x in lst.list() if isinstance(x, (int, float))]
    return Payload(min(nums), lst.confidence * 0.99, "min"), lst.confidence * 0.99 if nums else 0.0

def _max(lst: Payload):
    nums = [x for x in lst.list() if isinstance(x, (int, float))]
    return Payload(max(nums), lst.confidence * 0.99, "max"), lst.confidence * 0.99 if nums else 0.0

def _any(lst: Payload):
    bools = [x for x in lst.list() if isinstance(x, bool)]
    return Payload(any(bools), lst.confidence, "any"), lst.confidence

def _all(lst: Payload):
    bools = [x for x in lst.list() if isinstance(x, bool)]
    return Payload(all(bools), lst.confidence, "all"), lst.confidence

# --- Type conversion ---
def _to_string(p: Payload):
    return Payload(str(p.data), p.confidence * 0.99, "to_string"), p.confidence * 0.99

def _to_number(p: Payload):
    try:
        return Payload(float(p.data), p.confidence * 0.95, "to_number"), p.confidence * 0.95
    except (TypeError, ValueError):
        return Payload(None, 0.0, "convert_error"), 0.0

def _to_list(p: Payload):
    if p.is_list(): return p, p.confidence
    if p.is_dict(): return Payload(list(p.data.items()), p.confidence * 0.98, "to_list"), p.confidence * 0.98
    return Payload([p.data], p.confidence, "to_list"), p.confidence

def _type_of(p: Payload):
    return Payload(infer_type(p).value, 1.0, "type_of"), 1.0

# --- Logic ---
def _not(p: Payload):
    if isinstance(p.data, bool):
        return Payload(not p.data, p.confidence, "not"), p.confidence
    return Payload(None, 0.0, "not_bool"), 0.0

def _and(a: Payload, b: Payload):
    return Payload(bool(a.data) and bool(b.data), bayesian_combine(a.confidence, b.confidence), "and"), bayesian_combine(a.confidence, b.confidence)

def _or(a: Payload, b: Payload):
    return Payload(bool(a.data) or bool(b.data), bayesian_combine(a.confidence, b.confidence), "or"), bayesian_combine(a.confidence, b.confidence)

# Register all operations
N = PayloadType.NUMBER
S = PayloadType.STRING
L = PayloadType.LIST
D = PayloadType.DICT
B = PayloadType.BOOL
A = PayloadType.ANY

for name, fn, inputs, output in [
    ("add", _add, [N, N], N), ("sub", _sub, [N, N], N),
    ("mul", _mul, [N, N], N), ("div", _div, [N, N], N),
    ("pow", _pow, [N, N], N), ("mod", _mod, [N, N], N),
    ("eq", _eq, [A, A], B), ("neq", _neq, [A, A], B),
    ("gt", _gt, [N, N], B), ("gte", _gte, [N, N], B),
    ("lt", _lt, [N, N], B), ("lte", _lte, [N, N], B),
    ("sort_asc", _sort_asc, [L], L), ("sort_desc", _sort_desc, [L], L),
    ("reverse", _reverse, [L], L), ("first", _first, [L], A),
    ("last", _last, [L], A), ("rest", _rest, [L], L),
    ("count", _count, [L], N), ("flatten", _flatten, [L], L),
    ("unique", _unique, [L], L), ("slice", _slice, [L, N, N], L),
    ("zip", _zip, [L, L], L),
    ("upper", _upper, [S], S), ("lower", _lower, [S], S),
    ("strip", _strip, [S], S), ("split", _split, [S, S], L),
    ("join", _join, [L, S], S), ("replace", _replace, [S, S, S], S),
    ("len", _len, [A], N), ("contains", _contains, [A, A], B),
    ("starts_with", _starts_with, [S, S], B), ("ends_with", _ends_with, [S, S], B),
    ("get", _get, [D, A], A), ("keys", _keys, [D], L),
    ("values", _values, [D], L), ("merge", _merge, [D, D], D),
    ("sum", _sum, [L], N), ("mean", _mean, [L], N),
    ("min", _min, [L], N), ("max", _max, [L], N),
    ("any", _any, [L], B), ("all", _all, [L], B),
    ("to_string", _to_string, [A], S), ("to_number", _to_number, [A], N),
    ("to_list", _to_list, [A], L), ("type_of", _type_of, [A], S),
    ("not", _not, [B], B), ("and", _and, [A, A], B), ("or", _or, [A, A], B),
    ("filter", _filter, [L, D], L), ("map", _map, [L, S], L),
    ("reduce", _reduce, [L, S], A),
]:
    register_op(name, inputs, output, fn, desc=f"{name}({', '.join(i.value for i in inputs)}) -> {output.value}")


# ============================================================
# CONSIDER / RESOLVE / FORFEIT PROTOCOL
# ============================================================

class ConsiderResolveForfeit:
    """The three-state deliberation protocol.
    
    Consider: I see your proposal, here's my assessment
    Resolve: I accept/reject your proposal with reasoning
    Forfeit: I yield my approach to yours (with confidence transfer)
    """
    
    def __init__(self):
        self.considerations = []
        self.resolutions = []
        self.forfeits = []
    
    def consider(self, agent: str, proposal_id: str, assessment: str, 
                 confidence_adjustment: float = 0.0, concerns: list = None) -> dict:
        c = {
            "agent": agent, "proposal_id": proposal_id,
            "assessment": assessment, "confidence_adj": confidence_adjustment,
            "concerns": concerns or [], "timestamp": time.time(),
        }
        self.considerations.append(c)
        return c
    
    def resolve(self, agent: str, proposal_id: str, accepted: bool,
                reason: str, confidence: float = 1.0) -> dict:
        r = {
            "agent": agent, "proposal_id": proposal_id,
            "accepted": accepted, "reason": reason,
            "confidence": confidence, "timestamp": time.time(),
        }
        self.resolutions.append(r)
        return r
    
    def forfeit(self, from_agent: str, to_agent: str, proposal_id: str,
                confidence_transferred: float, reason: str) -> dict:
        f = {
            "from": from_agent, "to": to_agent, "proposal_id": proposal_id,
            "confidence_transferred": confidence_transferred,
            "reason": reason, "timestamp": time.time(),
        }
        self.forfeits.append(f)
        return f


# ============================================================
# DELIBERATION ENGINE — orchestrates multi-agent deliberation
# ============================================================

class DeliberationEngine:
    """Orchestrates agents through Consider/Resolve/Forfeit to produce artifacts."""
    
    def __init__(self, max_rounds: int = 10, confidence_threshold: float = 0.85,
                 convergence_window: int = 3):
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.convergence_window = convergence_window
        self.protocol = ConsiderResolveForfeit()
        self.rounds: list[DeliberationRound] = []
        self.artifacts: list[dict] = []
    
    def deliberate(self, intent: str, agents: list[dict] = None) -> dict:
        """Run full deliberation cycle on an intent.
        
        Args:
            intent: Natural language intention
            agents: Optional list of {"name": str, "role": AgentRole} dicts
        
        Returns:
            Deliberation result with output payload and trace
        """
        intent_payload = Payload(intent, 1.0, "user_intent")
        
        # Default agents
        if not agents:
            agents = [
                {"name": "intent_parser", "role": AgentRole.INTENT_PARSER},
                {"name": "architect", "role": AgentRole.ARCHITECT},
                {"name": "validator", "role": AgentRole.VALIDATOR},
                {"name": "optimizer", "role": AgentRole.OPTIMIZER},
                {"name": "codegen", "role": AgentRole.CODEGEN},
            ]
        
        confidence_history = []
        
        for round_num in range(1, self.max_rounds + 1):
            round_data = DeliberationRound(round_num, intent_payload)
            
            # Phase 1: Parse intent
            parsed = self._parse_intent(intent)
            
            # Phase 2: Propose operations
            proposals = self._propose_operations(parsed)
            
            # Phase 3: Validate proposals
            validated = self._validate_proposals(proposals)
            
            # Phase 4: Optimize
            optimized = self._optimize_proposals(validated)
            
            # Phase 5: Generate
            output = self._generate_output(parsed, optimized)
            
            round_data.output = output
            confidence_history.append(output.confidence)
            
            # Check convergence
            if self._check_convergence(confidence_history):
                round_data.converged = True
            
            self.rounds.append(round_data)
            
            if round_data.converged or output.confidence >= self.confidence_threshold:
                break
        
        # Build final artifact
        best_round = max(self.rounds, key=lambda r: r.output.confidence if r.output else 0)
        artifact = {
            "intent": intent,
            "output": best_round.output.to_dict() if best_round.output else None,
            "rounds": len(self.rounds),
            "converged": any(r.converged for r in self.rounds),
            "trace": self._build_trace(),
            "operations_used": self._extract_operations(best_round),
        }
        self.artifacts.append(artifact)
        return artifact
    
    def _parse_intent(self, intent: str) -> dict:
        """Parse a natural language intent into structured components."""
        intent_lower = intent.lower()
        
        # Extract data type
        data_type = "list"
        if "string" in intent_lower or "text" in intent_lower or "word" in intent_lower:
            data_type = "string"
        elif "dict" in intent_lower or "object" in intent_lower or "record" in intent_lower:
            data_type = "dict"
        elif "number" in intent_lower or "num" in intent_lower:
            data_type = "number"
        
        # Extract operations
        ops = []
        op_keywords = {
            "sort": "sort", "order": "sort", "arrange": "sort", "rank": "sort",
            "filter": "filter", "select": "filter", "keep": "filter", "only": "filter",
            "count": "count", "len": "count", "length": "count", "how many": "count",
            "sum": "sum", "total": "sum", "add up": "sum", "accumulate": "sum",
            "average": "mean", "mean": "mean", "avg": "mean",
            "min": "min", "minimum": "min", "smallest": "min", "lowest": "min",
            "max": "max", "maximum": "max", "largest": "max", "highest": "max",
            "unique": "unique", "distinct": "unique", "deduplicate": "unique",
            "reverse": "reverse", "flip": "reverse",
            "first": "first", "head": "first",
            "last": "last", "tail": "last",
            "split": "split", "separate": "split", "break": "split",
            "join": "join", "combine": "join", "concatenate": "join", "concat": "join",
            "upper": "upper", "uppercase": "upper",
            "lower": "lower", "lowercase": "lower",
            "strip": "strip", "trim": "strip",
            "contains": "contains", "has": "contains", "includes": "contains",
            "replace": "replace", "substitute": "replace",
            "flatten": "flatten",
            "group": "group", "group_by": "group_by",
            "map": "map", "transform": "map", "apply": "map",
            "reduce": "reduce", "fold": "reduce",
            "merge": "merge", "combine dict": "merge",
        }
        
        for keyword, op in op_keywords.items():
            if keyword in intent_lower and op not in ops:
                ops.append(op)
        
        # Extract sort direction
        sort_direction = "asc"
        if any(w in intent_lower for w in ["descending", "desc", "highest first", "largest first", "reverse order"]):
            sort_direction = "desc"
        
        # Extract filter conditions
        conditions = {}
        # "greater than X", "less than X", "above X", "below X"
        for pattern, key in [("greater than (\\d+)", "gt"), ("less than (\\d+)", "lt"),
                             ("above (\\d+)", "gt"), ("below (\\d+)", "lt"),
                             ("over (\\d+)", "gt"), ("under (\\d+)", "lt")]:
            m = re.search(pattern, intent_lower)
            if m:
                conditions[key] = float(m.group(1))
        
        return {
            "raw": intent, "data_type": data_type, "operations": ops,
            "sort_direction": sort_direction, "conditions": conditions,
        }
    
    def _propose_operations(self, parsed: dict) -> list[dict]:
        proposals = []
        # Resolve parsed op names to registered ops
        op_alias = {
            "sort": "sort_desc" if parsed.get("sort_direction") == "desc" else "sort_asc",
        }
        for op_name in parsed["operations"]:
            resolved = op_alias.get(op_name, op_name)
            if resolved in OPERATIONS:
                op = OPERATIONS[resolved]
                proposals.append({
                    "name": resolved, "inputs": [i.value for i in op.inputs],
                    "output": op.output.value, "cost": op.cost,
                    "confidence": 0.9, "description": op.description,
                })
        return proposals
    
    def _validate_proposals(self, proposals: list[dict]) -> list[dict]:
        validated = []
        for p in proposals:
            errors = []
            if p["name"] not in OPERATIONS:
                errors.append("unknown_operation")
            if not errors:
                p["validated"] = True
                p["confidence"] = min(p["confidence"] + 0.05, 1.0)
            else:
                p["validated"] = False
                p["confidence"] = max(p["confidence"] - 0.3, 0.0)
                p["errors"] = errors
            validated.append(p)
        return validated
    
    def _optimize_proposals(self, proposals: list[dict]) -> list[dict]:
        """Optimize by removing redundant operations and ordering correctly."""
        optimized = []
        seen_ops = set()
        
        # Topological sort of operations
        op_order = ["filter", "map", "sort_asc", "sort_desc", "reverse", "unique",
                    "flatten", "slice", "first", "last", "rest", "count",
                    "sum", "mean", "min", "max", "join", "len"]
        
        for op_name in op_order:
            for p in proposals:
                if p["name"] == op_name and op_name not in seen_ops:
                    optimized.append(p)
                    seen_ops.add(op_name)
        
        # Add remaining ops
        for p in proposals:
            if p["name"] not in seen_ops:
                optimized.append(p)
                seen_ops.add(p["name"])
        
        return optimized
    
    def _generate_output(self, parsed: dict, proposals: list[dict]) -> Payload:
        """Generate the output payload from parsed intent and proposals."""
        ops_used = [p["name"] for p in proposals if p.get("validated", True)]
        
        # Build a function chain description
        chain = " -> ".join(ops_used) if ops_used else "identity"
        
        # Build executable Python code from the operation chain
        code = self._build_executable(parsed, ops_used)
        
        # Calculate confidence based on how well we understood the intent
        base_conf = 0.7
        if parsed["operations"]:
            base_conf += min(len(parsed["operations"]) * 0.05, 0.2)
        if parsed["conditions"]:
            base_conf += 0.05
        conf = min(base_conf, 0.95)
        
        return Payload({
            "chain": chain,
            "operations": ops_used,
            "code": code,
            "data_type": parsed["data_type"],
            "conditions": parsed["conditions"],
            "sort_direction": parsed["sort_direction"],
        }, conf, "codegen")
    
    def _build_executable(self, parsed: dict, ops: list[str]) -> str:
        """Build executable Python code from operation chain."""
        lines = ["def resolve(data):"]
        
        if not ops:
            lines.append("    return data")
            return "\n".join(lines)
        
        # Determine input variable
        var = "data"
        
        for op in ops:
            if op == "sort_asc":
                lines.append(f"    {var} = sorted({var})")
            elif op == "sort_desc":
                lines.append(f"    {var} = sorted({var}, reverse=True)")
            elif op == "reverse":
                lines.append(f"    {var} = list(reversed({var}))")
            elif op == "filter":
                conds = parsed.get("conditions", {})
                cond_strs = []
                if "gt" in conds:
                    cond_strs.append(f"x > {conds['gt']}")
                if "lt" in conds:
                    cond_strs.append(f"x < {conds['lt']}")
                if cond_strs:
                    combined = " and ".join(cond_strs)
                    lines.append(f"    {var} = [x for x in {var} if {combined}]")
                else:
                    lines.append(f"    {var} = [x for x in {var} if x]")
            elif op == "map":
                lines.append(f"    {var} = [transform(x) for x in {var}]")
            elif op == "unique":
                lines.append(f"    {var} = list(dict.fromkeys({var}))")
            elif op == "flatten":
                lines.append(f"    {var} = [item for sublist in {var} for item in (sublist if isinstance(sublist, list) else [sublist])]")
            elif op == "count":
                lines.append(f"    {var} = len({var})")
            elif op == "sum":
                lines.append(f"    {var} = sum({var})")
            elif op == "mean":
                lines.append(f"    {var} = sum({var}) / len({var}) if {var} else 0")
            elif op == "min":
                lines.append(f"    {var} = min({var}) if {var} else None")
            elif op == "max":
                lines.append(f"    {var} = max({var}) if {var} else None")
            elif op == "first":
                lines.append(f"    {var} = {var}[0] if {var} else None")
            elif op == "last":
                lines.append(f"    {var} = {var}[-1] if {var} else None")
            elif op == "rest":
                lines.append(f"    {var} = {var}[1:] if {var} else []")
            elif op == "upper":
                lines.append(f"    {var} = {var}.upper()")
            elif op == "lower":
                lines.append(f"    {var} = {var}.lower()")
            elif op == "strip":
                lines.append(f"    {var} = {var}.strip()")
            elif op == "split":
                lines.append(f"    {var} = {var}.split()")
            elif op == "join":
                lines.append(f"    {var} = ' '.join(str(x) for x in {var})")
            elif op == "len":
                lines.append(f"    {var} = len({var})")
            elif op == "contains":
                lines.append(f"    # contains: check if target in {var}")
            elif op == "reverse" and parsed["data_type"] == "string":
                lines.append(f"    {var} = {var}[::-1]")
            elif op == "slice":
                lines.append(f"    {var} = {var}[1:-1]")
            elif op == "merge":
                lines.append(f"    {var} = {{**{var}, **other_dict}}")
        
        lines.append(f"    return {var}")
        return "\n".join(lines)
    
    def _check_convergence(self, history: list[float]) -> bool:
        if len(history) < self.convergence_window:
            return False
        recent = history[-self.convergence_window:]
        variance = sum((c - sum(recent)/len(recent))**2 for c in recent) / len(recent)
        return variance < 0.01
    
    def _build_trace(self) -> list[dict]:
        trace = []
        for r in self.rounds:
            trace.append({
                "round": r.round_num,
                "confidence": r.output.confidence if r.output else 0.0,
                "converged": r.converged,
                "proposals": len(r.proposals),
            })
        return trace
    
    def _extract_operations(self, round_data: DeliberationRound) -> list[str]:
        if round_data.output and round_data.output.is_dict():
            return round_data.output.data.get("operations", [])
        return []


# ============================================================
# .rsn FILE PARSER — compile resolve source files
# ============================================================

class RsnParser:
    """Parse .rsn resolve source files."""
    
    def __init__(self):
        self.variables: dict[str, Payload] = {}
        self.operations_order: list[str] = []
    
    def parse(self, source: str) -> dict:
        """Parse a .rsn source file into executable steps."""
        lines = source.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            step = self._parse_line(line)
            if step:
                steps.append(step)
        return {"steps": steps, "variables": dict(self.variables)}
    
    def _parse_line(self, line: str) -> Optional[dict]:
        # Assignment: let x = operation(args)
        m = re.match(r'let\s+(\w+)\s*=\s*(.+)', line)
        if m:
            var_name = m.group(1)
            expr = m.group(2).strip()
            return {"type": "let", "name": var_name, "expr": expr}
        
        # Operation call: operation(arg1, arg2)
        m = re.match(r'(\w+)\((.+)\)', line)
        if m and m.group(1) in OPERATIONS:
            op_name = m.group(1)
            args = [a.strip() for a in m.group(2).split(",")]
            return {"type": "call", "operation": op_name, "args": args}
        
        # Intent: intent "description"
        m = re.match(r'intent\s+"(.+)"', line)
        if m:
            return {"type": "intent", "text": m.group(1)}
        
        # Resolve: resolve "description"
        m = re.match(r'resolve\s+"(.+)"', line)
        if m:
            return {"type": "resolve", "text": m.group(1)}
        
        # Constraint: constraint "description"
        m = re.match(r'constraint\s+"(.+)"', line)
        if m:
            return {"type": "constraint", "text": m.group(1)}
        
        # Confidence: confidence 0.9
        m = re.match(r'confidence\s+([\d.]+)', line)
        if m:
            return {"type": "confidence", "value": float(m.group(1))}
        
        return None
    
    def execute(self, source: str) -> Payload:
        """Parse and execute a .rsn program, return final output."""
        parsed = self.parse(source)
        last_result = Payload(None, 0.0)
        
        for step in parsed["steps"]:
            if step["type"] == "let":
                result = self._eval_expr(step["expr"], self.variables)
                self.variables[step["name"]] = result
                last_result = result
            elif step["type"] == "call":
                args = [self._eval_expr(a, self.variables) for a in step["args"]]
                op = OPERATIONS[step["operation"]]
                result = op.fn(*args)[0]
                last_result = result
            elif step["type"] == "resolve":
                engine = DeliberationEngine()
                artifact = engine.deliberate(step["text"])
                last_result = Payload(artifact, artifact["output"]["confidence"] if artifact["output"] else 0.5, "resolve")
        
        return last_result
    
    def _eval_expr(self, expr: str, variables: dict) -> Payload:
        """Evaluate an expression to a Payload."""
        expr = expr.strip()
        
        # Number literal
        try:
            return Payload(float(expr), 1.0, "literal")
        except ValueError:
            pass
        
        try:
            return Payload(int(expr), 1.0, "literal")
        except ValueError:
            pass
        
        # String literal
        if expr.startswith('"') and expr.endswith('"'):
            return Payload(expr[1:-1], 1.0, "literal")
        
        # List literal
        if expr.startswith("[") and expr.endswith("]"):
            try:
                inner = expr[1:-1]
                if not inner.strip():
                    return Payload([], 1.0, "literal")
                items = [self._eval_expr(i.strip(), variables).data for i in inner.split(",")]
                return Payload(items, 0.95, "literal")
            except:
                pass
        
        # Variable reference
        if expr in variables:
            return variables[expr]
        
        # Operation call
        m = re.match(r'(\w+)\((.+)\)', expr)
        if m and m.group(1) in OPERATIONS:
            args = [self._eval_expr(a.strip(), variables) for a in m.group(2).split(",")]
            return OPERATIONS[m.group(1)].fn(*args)[0]
        
        return Payload(expr, 0.5, "unknown")


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Resolve — Functional A2A Deliberative Compilation")
        print()
        print("Usage:")
        print("  resolve \"sort numbers descending\"     # Natural language intent")
        print("  resolve build program.rsn              # Build a .rsn file")
        print("  resolve run program.rsn                # Execute a .rsn file")
        print("  resolve ops                            # List available operations")
        print("  resolve eval 'add(3, 4)'               # Evaluate an expression")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "ops":
        print(f"Available operations ({len(OPERATIONS)}):")
        for name in sorted(OPERATIONS.keys()):
            op = OPERATIONS[name]
            print(f"  {name:20s} {op.description}")
    
    elif command == "eval":
        if len(sys.argv) < 3:
            print("Usage: resolve eval 'expression'")
            sys.exit(1)
        parser = RsnParser()
        result = parser._eval_expr(sys.argv[2], {})
        print(f"Result: {result}")
    
    elif command == "build":
        if len(sys.argv) < 3:
            print("Usage: resolve build program.rsn")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            source = f.read()
        parser = RsnParser()
        parsed = parser.parse(source)
        print(f"Parsed {len(parsed['steps'])} steps")
        for i, step in enumerate(parsed["steps"]):
            print(f"  {i+1}. {step}")
    
    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: resolve run program.rsn")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            source = f.read()
        parser = RsnParser()
        result = parser.execute(source)
        print(f"Output: {result}")
    
    else:
        # Natural language intent
        engine = DeliberationEngine()
        artifact = engine.deliberate(" ".join(sys.argv[1:]))
        
        output = artifact.get("output", {})
        if output:
            print(f"Confidence: {output['confidence']:.1%}")
            print(f"Operations: {output.get('operations', [])}")
            print(f"Chain: {output.get('chain', '')}")
            print()
            print("Generated code:")
            print(output.get("code", ""))
            print()
            print(f"Rounds: {artifact['rounds']} | Converged: {artifact['converged']}")
        else:
            print("Failed to resolve intent")


if __name__ == "__main__":
    main()
