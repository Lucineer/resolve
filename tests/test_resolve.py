"""Tests for Resolve A2A Deliberation System."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resolve import (
    Payload, PayloadChain, Intent, IntentParser, IntentParserAgent,
    ArchitectAgent, ValidatorAgent, OptimizerAgent, MetaDeliberator,
    DeliberationEngine, TraceEntry, resolve, Artifact
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")

print("=== Payload Tests ===")

# Basic payload creation
p = Payload(op="test", confidence=0.8)
test("payload has op", p.op == "test")
test("payload has confidence", p.confidence == 0.8)
test("payload has id", len(p._id) == 8)

# JSON round-trip
j = p.to_json()
p2 = Payload.from_json(j)
test("json round-trip op", p2.op == "test")
test("json round-trip confidence", abs(p2.confidence - 0.8) < 0.001)
test("json round-trip id", p2._id == p._id)

# Clone
c = p.clone()
test("clone has different id", c._id != p._id)
test("clone has same op", c.op == p.op)
test("clone is independent", c.confidence != 0.42)
c.confidence = 0.42
test("original unchanged after clone", p.confidence == 0.8)

# Bayesian confidence merge
a = Payload(confidence=0.5)
b = Payload(confidence=0.5)
combined = a.merge_confidence(b)
# 1/(1/0.5 + 1/0.5) = 1/4 = 0.25
test("bayesian merge 0.5+0.5", abs(combined - 0.25) < 0.001)

a2 = Payload(confidence=0.9)
b2 = Payload(confidence=0.9)
c2 = a2.merge_confidence(b2)
# 1/(1/0.9 + 1/0.9) = 0.45
test("bayesian merge 0.9+0.9", abs(c2 - 0.45) < 0.001)

# Edge case: zero confidence
z = Payload(confidence=0.0)
nz = Payload(confidence=0.5)
z_comb = z.merge_confidence(nz)
test("zero confidence handled", 0 <= z_comb < 0.01)

print("\n=== PayloadChain Tests ===")

chain = PayloadChain()
test("empty chain agg conf", chain.aggregate_confidence() == 0.0)
test("empty chain best", chain.best() is None)

chain.add(Payload(confidence=0.5))
chain.add(Payload(confidence=0.8))
chain.add(Payload(confidence=0.3))
test("chain size", len(chain.chain) == 3)
test("chain best", chain.best().confidence == 0.8)
test("chain aggregate", abs(chain.aggregate_confidence() - 0.533) < 0.05)

dag = chain.get_dag()
test("dag returns list", len(dag) == 3)
test("dag has id", "id" in dag[0])

print("\n=== Intent Tests ===")

intent = Intent(goal="sort numbers descending", hard_constraints=["descending"])
test("intent goal", intent.goal == "sort numbers descending")
test("intent hard constraints", "descending" in intent.hard_constraints)

p_intent = intent.to_payload()
test("intent to_payload op", p_intent.op == "intent")
test("intent to_payload confidence", p_intent.confidence == 1.0)

parser = IntentParser()
parsed = parser.extract("sort a list of numbers in descending order")
test("parser extracts descending", any("descending" in c for c in parsed.hard_constraints))
test("parser extracts list", any("list" in c for c in parsed.hard_constraints))

parsed2 = parser.extract("sort items efficiently with error handling")
test("parser extracts efficient", any("efficient" in c for c in parsed2.soft_constraints))
test("parser extracts error handling", any("edge case" in c for c in parsed2.soft_constraints))

print("\n=== Agent Tests ===")

# IntentParserAgent
ipa = IntentParserAgent()
result = ipa.receive(Payload(op="intent", inputs=["sort numbers descending"]))
test("intent parser agent role", ipa.role == "intent_parser")
test("intent parser returns parsed_intent", result.op == "parsed_intent")
test("intent parser confidence > 0.9", result.confidence > 0.9)
test("intent parser tracks proposals", ipa.proposals_made == 1)

# ArchitectAgent
arch = ArchitectAgent()
result2 = arch.receive(Payload(op="intent", inputs=["sort a list"]))
test("architect returns proposal", result2.op == "propose_architecture")
test("architect has code", "sorted" in result2.metadata.get("result", {}).get("code", ""))
test("architect tracks proposals", arch.proposals_made == 1)

# ValidatorAgent
val = ValidatorAgent()
arch_proposal = Payload(
    op="propose_architecture", confidence=0.9,
    constraints={"hard": ["descending"]},
    metadata={"result": {"name": "builtin sorted", "code": "sorted(data, reverse=True)"}}
)
vresult = val.receive(arch_proposal)
test("validator returns validation", vresult.op == "validation")
test("validator passed descending", vresult.metadata["result"].get("passed", False) == True)
test("validator confidence > 0.8", vresult.confidence > 0.8)

# Failing validation
bad_proposal = Payload(
    op="propose_architecture", confidence=0.7,
    constraints={"hard": ["descending"]},
    metadata={"result": {"name": "bad", "code": "sorted(data)"}}
)
vbad = val.receive(bad_proposal)
test("validator catches wrong order", vbad.metadata["result"].get("passed", True) == False)
test("validator adjusts confidence on failure", 0.5 <= vbad.confidence <= 0.8)

# OptimizerAgent
opt = OptimizerAgent()
opt_result = opt.receive(arch_proposal)
test("optimizer returns optimization", opt_result.op == "propose_optimization")
test("optimizer found optimizations", len(opt_result.metadata.get("result", [])) > 0)

# MetaDeliberator
meta = MetaDeliberator(0.85)
conv = meta.receive(Payload(op="convergence_check", confidence=0.9))
test("meta converges at 0.9", conv.metadata["result"]["converged"] == True)
no_conv = meta.receive(Payload(op="convergence_check", confidence=0.5))
test("meta doesn't converge at 0.5", no_conv.metadata["result"]["converged"] == False)

print("\n=== Deliberation Engine Tests ===")

engine = DeliberationEngine(verbose=False, confidence_threshold=0.8, max_rounds=5)
artifact = engine.deliberate("sort a list of numbers in descending order")

test("engine produces artifact", artifact is not None)
test("artifact has code", len(artifact.result_code) > 0)
test("artifact has descending", "reverse=True" in artifact.result_code)
test("artifact has confidence", artifact.confidence > 0.0)
test("artifact has agents", len(artifact.agents_involved) >= 3)
test("artifact has provenance", len(artifact.provenance_chain) > 0)
test("engine has trace", len(engine.trace) > 0)

print("\n=== Executable Artifact Tests ===")

fn = artifact.to_executable()
test("artifact is callable", callable(fn))
result = fn([3, 1, 4, 1, 5, 9, 2, 6])
test("sort descending correct", result == [9, 6, 5, 4, 3, 2, 1, 1])

# Test with strings
fn2 = artifact.to_executable()
result2 = fn2(["banana", "apple", "cherry"])
test("sort strings descending", result2 == ["cherry", "banana", "apple"])

# Test trace output
trace_str = artifact.trace()
test("trace has DELIBERATION TRACE", "DELIBERATION TRACE" in trace_str)
test("trace has agents", "intent_parser" in trace_str)

print("\n=== Resolve() Main Function Tests ===")

a2 = resolve("sort a list of numbers in descending order", verbose=False)
test("resolve returns artifact", isinstance(a2, Artifact))
fn3 = a2.to_executable()
r3 = fn3([10, 5, 8, 3])
test("resolve sort works", r3 == [10, 8, 5, 3])

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"{failed} FAILURES ✗")
print(f"{'='*60}")
