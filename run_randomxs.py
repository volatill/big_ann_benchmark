#!/usr/bin/env python3
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

DATASETS = os.environ.get("BABB_DATASETS", "random-xs").split(",")
RUN_GROUP = os.environ.get("BABB_RUN_GROUP", "base")

FAISS_ALGO = os.environ.get("BABB_FAISS_ALGO", "buddy-t1")
LSMVEC_ALGO = os.environ.get("BABB_LSMVEC_ALGO", "lsmvec")

EXTRA_ARGS = os.environ.get("BABB_EXTRA_ARGS", "").strip()

def runCmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)

def captureCmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout

def findFileUpToDepth(root: Path, name: str, maxDepth: int = 4) -> Optional[Path]:
    candidates = [root]
    for _ in range(maxDepth + 1):
        nextCandidates = []
        for d in candidates:
            f = d / name
            if f.exists():
                return f
            for child in d.iterdir():
                if child.is_dir() and child.name not in [".git", "__pycache__", ".venv", "venv", "build", "dist"]:
                    nextCandidates.append(child)
        candidates = nextCandidates
    return None

def pickFlag(helpText: str, options: List[str]) -> Optional[str]:
    for opt in options:
        if re.search(rf"(^|\s){re.escape(opt)}(\s|,|=|$)", helpText):
            return opt
    return None

def detectRunner(repoRoot: Path) -> Tuple[List[str], str]:
    candidates: List[List[str]] = []
    if (repoRoot / "run.py").exists():
        candidates.append(["python3", "run.py"])
    if (repoRoot / "benchmark" / "run.py").exists():
        candidates.append(["python3", str(repoRoot / "benchmark" / "run.py")])
    candidates.append(["python3", "-m", "benchmark.run"])
    candidates.append(["python3", "-m", "benchmark.runner"])
    candidates.append(["python3", "-m", "benchmark.run_benchmark"])

    for baseCmd in candidates:
        rc, out = captureCmd(baseCmd + ["--help"], cwd=repoRoot)
        if rc == 0 and ("usage:" in out.lower() or "--" in out):
            return baseCmd, out

    raise RuntimeError("Could not find a working benchmark runner entrypoint.")

def buildRunCommand(
    baseCmd: List[str],
    helpText: str,
    dataset: str,
    algorithm: str,
    runGroup: str,
    algosYaml: Optional[Path],
) -> List[str]:
    datasetFlag = pickFlag(helpText, ["--dataset", "--dataset-name", "--dataset_name"])
    algoFlag = pickFlag(helpText, ["--algorithm", "--algo", "--algorithm-name", "--algorithm_name"])
    groupFlag = pickFlag(helpText, ["--run-group", "--run_group", "--group", "--runGroup"])
    configFlag = pickFlag(helpText, ["--definitions", "--algos", "--algos-yaml", "--algos_yaml", "--config", "--configuration"])

    if datasetFlag is None or algoFlag is None:
        raise RuntimeError("Runner flags not recognized from --help output.")

    cmd = list(baseCmd)
    cmd += [datasetFlag, dataset, algoFlag, algorithm]

    if groupFlag is not None:
        cmd += [groupFlag, runGroup]

    if configFlag is not None and algosYaml is not None:
        cmd += [configFlag, str(algosYaml)]

    if EXTRA_ARGS:
        cmd += EXTRA_ARGS.split()

    return cmd

def main() -> int:
    repoRoot = Path.cwd()
    algosYaml = findFileUpToDepth(repoRoot, "algos-2021.yaml", maxDepth=5)

    baseCmd, helpText = detectRunner(repoRoot)
    print("Detected benchmark runner:", " ".join(baseCmd))
    if algosYaml:
        print("Found algos config:", algosYaml)

    jobs: List[Tuple[str, str]] = []
    for ds in DATASETS:
        ds = ds.strip()
        if not ds:
            continue
        jobs.append((ds, FAISS_ALGO))
        jobs.append((ds, LSMVEC_ALGO))

    for ds, algo in jobs:
        cmd = buildRunCommand(baseCmd, helpText, ds, algo, RUN_GROUP, algosYaml)
        runCmd(cmd, cwd=repoRoot, check=False)

    print("\nAll requested runs finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
