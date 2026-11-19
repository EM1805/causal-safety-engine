"""Resource Genesis v0.1: CPU / RSS / Python-allocation measurements.

Separate Python subprocesses for each algorithm+workload; same frozen source,
same synthetic graph, same outputs. This is not end-to-end app profiling.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, resource, statistics
import subprocess, sys, time, tracemalloc, types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE="08b2e17bf5d00d165909390f21293a5c18515e7d"
MODES=("original","deque_only","reverse_hybrid")
SHAPES=("wide_unreachable","wide_reachable","wide_mixed","narrow_reachable")

def load(mode):
    if mode=="reverse_hybrid":
        spec=importlib.util.spec_from_file_location("candidate",ROOT/"tools/do_readiness.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    cp=subprocess.run(["git","show",f"{BASE}:tools/do_readiness.py"],
                      cwd=ROOT,capture_output=True,text=True,check=True)
    module=types.ModuleType("frozen")
    exec(compile(cp.stdout,"frozen_native","exec"),module.__dict__)
    if mode=="deque_only":
        from collections import deque
        def path(G,src,dst,blocked):
            if src in blocked or dst in blocked:return False
            queue=deque([src]);seen={src}
            while queue:
                u=queue.popleft()
                if u==dst:return True
                for v in G.successors(u):
                    if v not in blocked and v not in seen:
                        seen.add(v);queue.append(v)
            return False
        module.has_path_excluding=path
    return module

def graph(module,shape):
    if shape not in SHAPES:raise ValueError(shape)
    parents=2 if shape=="narrow_reachable" else 160
    g=module.DiGraph()
    g.add_node("X");g.add_node("Y")
    for j in range(parents):
        g.add_edge(f"p{j}","X");g.add_edge(f"p{j}",f"v{j%12}")
    for j in range(799):g.add_edge(f"v{j}",f"v{j+1}")
    if shape in ("wide_reachable","narrow_reachable"):
        g.add_edge("v799","Y")
    if shape=="wide_mixed":
        for j in range(0,parents,5):g.add_edge(f"p{j}","Y")
    return g

def worker(mode,shape,repeats):
    mod=load(mode);g=graph(mod,shape)
    expected=mod.backdoor_confounders(g,"X","Y")
    start=time.process_time_ns()
    for _ in range(repeats):
        if mod.backdoor_confounders(g,"X","Y")!=expected:
            raise AssertionError("unstable output")
    cpu=(time.process_time_ns()-start)*1e-9/repeats
    tracemalloc.start()
    tracemalloc.reset_peak()
    check=mod.backdoor_confounders(g,"X","Y")
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if check!=expected:raise AssertionError("instrumentation changed result")
    return dict(mode=mode,shape=shape,cpu_seconds_per_query=cpu,
                python_alloc_peak_bytes=peak,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                graph_nodes=len(g.nodes),n_conf=len(expected),
                output_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest())

def campaign(samples,repeats):
    receipts=[]
    for sample in range(samples):
        for shape in SHAPES:
            for mode in MODES[sample%3:]+MODES[:sample%3]:
                cp=subprocess.run([
                    sys.executable,"-m","resource_genesis.benchmark","--worker",
                    "--mode",mode,"--shape",shape,"--repeats",str(repeats)],
                    cwd=ROOT,capture_output=True,text=True,check=True,
                    env={**os.environ,"PYTHONHASHSEED":"0"})
                receipts.append(json.loads(cp.stdout))
    metrics={}
    for shape in SHAPES:
        subset=[r for r in receipts if r["shape"]==shape]
        if len(set(r["output_sha256"] for r in subset))!=1:
            raise AssertionError("baseline output differs from candidate")
        metrics[shape]={}
        for mode in MODES:
            selected=[r for r in subset if r["mode"]==mode]
            metrics[shape][mode]={
                "cpu_seconds_per_query":statistics.median(r["cpu_seconds_per_query"] for r in selected),
                "rss_peak_kib":statistics.median(r["rss_peak_kib"] for r in selected),
                "python_alloc_peak_bytes":statistics.median(r["python_alloc_peak_bytes"] for r in selected),
            }
        baseline=metrics[shape]["original"]
        for mode in MODES:
            row=metrics[shape][mode]
            for label in ("cpu_seconds_per_query","rss_peak_kib","python_alloc_peak_bytes"):
                row[label+"_ratio"]=row[label]/max(1e-12,baseline[label])
    cpu_gain=all(metrics[s]["reverse_hybrid"]["cpu_seconds_per_query_ratio"]<1 for s in SHAPES)
    rss_guard=all(metrics[s]["reverse_hybrid"]["rss_peak_kib_ratio"]<=1.10 for s in SHAPES)
    # Candidate selection never uses test labels; the set of algorithms is host-defined.
    return {
        "protocol":"Amantia Resource Genesis 0.1",
        "status":"REVIEW_CANDIDATE" if cpu_gain and rss_guard else "NOT_PROMOTED_OR_INCONCLUSIVE",
        "baseline_commit":BASE, "samples":samples,"repeats":repeats,
        "metrics":metrics,"receipts":receipts,
        "algorithm_candidates":["original","fixed deque BFS","reverse reachability hybrid"],
        "measurement_methods":"process_time_ns, Linux maxrss (whole subprocess), Python tracemalloc peak",
        "limitations":[
            "All graphs are authored synthetic workloads, not real application benchmarks.",
            "RSS includes Python startup, module imports and graph memory.",
            "Measured temporary allocations are not equivalent to resident RAM.",
            "A speed improvement need not imply lower RSS on every platform.",
            "Operator strategies are host-designed, not autonomous inventions.",
            "Graph-reachability semantics are not proof of causal identification."
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--mode",choices=MODES,default="original")
    p.add_argument("--shape",choices=SHAPES,default="wide_reachable")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--samples",type=int,default=3)
    p.add_argument("--output",default="")
    a=p.parse_args()
    if not 1<=a.repeats<=20 or not 1<=a.samples<=10:raise ValueError("budget")
    if a.worker:print(json.dumps(worker(a.mode,a.shape,a.repeats)));return
    report=campaign(a.samples,a.repeats)
    payload=json.dumps(report,sort_keys=True,indent=2)+"\n"
    if a.output:
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload,encoding="utf-8")
    print(json.dumps({"status":report["status"],"metrics":report["metrics"],
          "report_sha256":hashlib.sha256(payload.encode()).hexdigest()},
          sort_keys=True,indent=2))
if __name__=="__main__":main()
