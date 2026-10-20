"""Generate tests/fixtures/grade_3839a57_features.json from GRADE at commit 3839a57.

The library's feature layer is a port of GRADE's; tests/test_featurize.py compares it
with this fixture element by element, so the core tests do not need GRADE.

Usage: python benchmarks/catchbench_parity/gen_grade_fixture.py <grade checkout at 3839a57> <out.json>
"""
import json
import random
import subprocess
import sys

GRADE = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, GRADE + "/experiment")
sys.path.insert(0, GRADE + "/src")

head = subprocess.run(["git", "-C", GRADE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head.startswith("3839a57"), head

import grade  # noqa: E402
from agent_failure_localization import _step_matrix  # noqa: E402

assert grade.__file__.startswith(GRADE), grade.__file__

LAYERS = ["flat", "exec", "full", "flatdens", "flatdep", "depnorm"]
rng = random.Random(20261019)
names, cases = {}, []
for trial in range(24):
    n = rng.randint(2, 18)
    agents = [f"a{rng.randint(0, 3)}" for _ in range(n)]
    steps = []
    for i in range(n):
        deps = sorted(rng.sample(range(i), k=rng.randint(0, min(i, 3)))) if i else []
        steps.append({"idx": i, "agent": agents[i], "kind": rng.choice(["decision", "tool_call"]), "deps": deps})
    values = {}
    for mode in ("full_context", "chain", "explicit"):
        for shared in (True, False):
            G = grade.build_graph(steps, dependency=mode, shared_resource=shared)
            for layer in LAYERS:
                layer_names, layer_values = grade.feature_vector(G, layer=layer)
                assert names.setdefault(layer, layer_names) == layer_names
                values[f"{mode}/{int(shared)}/{layer}"] = layer_values
    G = grade.build_graph(steps, dependency="full_context")
    ww8 = _step_matrix({"steps": steps, "G": G}).tolist()
    cases.append({"steps": steps, "layers": values, "who_when_step_matrix": ww8})

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"source": "GRADE (https://github.com/yzhao062/grade, MIT License) at commit " + head,
               "generator": "benchmarks/catchbench_parity/gen_grade_fixture.py",
               "names": names, "cases": cases}, fh, separators=(",", ":"))
print(len(cases), "cases written to", OUT)
