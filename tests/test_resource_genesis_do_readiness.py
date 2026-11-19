"""Resource Genesis v0.1: frozen-original semantic equivalence checks.

No numerical claim is accepted on speed or RAM unless this comparison passes.
The native original is fetched from the exact pre-PR Git commit, without
editing or importing performance implementations from the modified branch.
"""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "08b2e17bf5d00d165909390f21293a5c18515e7d"


def load_frozen_original():
    cp = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:tools/do_readiness.py"],
        cwd=ROOT, capture_output=True, text=True, check=True
    )
    mod = types.ModuleType("frozen_do_readiness")
    exec(compile(cp.stdout, f"git:{BASE_COMMIT}/do_readiness.py", "exec"),
         mod.__dict__)
    return mod


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "candidate_do_readiness", ROOT / "tools" / "do_readiness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph(module, seed, n=75, edges=240):
    rng = random.Random(seed)
    g = module.DiGraph()
    for i in range(n):
        g.add_node(f"v{i}")
    for _ in range(edges):
        i, j = rng.randrange(n), rng.randrange(n)
        g.add_edge(f"v{i}", f"v{j}")
    return g


class ResourceGenesisParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_frozen_original()
        cls.candidate = load_candidate()

    def test_identifiability_matches_on_random_graphs(self):
        for seed in (5, 9, 21, 42):
            g1 = graph(self.original, seed)
            g2 = graph(self.candidate, seed)
            for x in range(0, 75, 7):
                for y in range(1, 75, 9):
                    start, end = f"v{x}", f"v{y}"
                    self.assertEqual(
                        self.original.backdoor_confounders(g1, start, end),
                        self.candidate.backdoor_confounders(g2, start, end),
                        (seed, start, end)
                    )
                    self.assertEqual(
                        self.original.check_identifiability(g1, start, end),
                        self.candidate.check_identifiability(g2, start, end)
                    )

    def test_has_path_excluding_matches_on_cyclic_and_sparse_graphs(self):
        for seed in (11, 14, 91):
            g1, g2 = graph(self.original, seed), graph(self.candidate, seed)
            for x in range(0, 75, 4):
                for y in range(0, 75, 13):
                    blocked={f"v{(x+y)%75}"}
                    self.assertEqual(
                        self.original.has_path_excluding(
                            g1, f"v{x}", f"v{y}", blocked
                        ),
                        self.candidate.has_path_excluding(
                            g2, f"v{x}", f"v{y}", blocked
                        ),
                    )

    def test_large_shared_ancestor_and_negative_controls(self):
        for module in (self.original, self.candidate):
            g=module.DiGraph()
            g.add_node("X"); g.add_node("Y")
            for i in range(300):
                g.add_edge(f"p{i}","X")
                g.add_edge(f"p{i}", f"chain{i%12}")
            for i in range(11):
                g.add_edge(f"chain{i}", f"chain{i+1}")
            g.add_edge("chain11","Y")
            self.assertEqual(
                module.backdoor_confounders(g, "X", "Y"),
                sorted([f"p{i}" for i in range(300)])
            )
            self.assertEqual(module.backdoor_confounders(g,"X","X"), [])
            g.add_edge("Y", "X")
            self.assertNotIn("Y",module.backdoor_confounders(g,"X","Y"))
            g.add_edge("X", "Y")
            self.assertNotIn("X",module.backdoor_confounders(g,"X","Y"))

    def test_edgecases_missing_nodes_and_direct_backdoor(self):
        for mod in (self.original,self.candidate):
            g=mod.DiGraph()
            g.add_node("X");g.add_node("Y")
            g.add_edge("Z","X");g.add_edge("Z","Y")
            self.assertEqual(mod.backdoor_confounders(g,"X","Y"), ["Z"])
            self.assertEqual(mod.backdoor_confounders(g,"missing","Y"), [])
            self.assertEqual(mod.check_identifiability(g,"X","X")["identifiable"],False)
            self.assertEqual(mod.check_identifiability(g,"X","missing")["identifiable"],False)
            g.add_edge("X","Z")
            self.assertEqual(mod.backdoor_confounders(g,"X","Y"), ["Z"])


if __name__=="__main__":
    unittest.main()
