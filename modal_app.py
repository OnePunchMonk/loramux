"""Run tests and benchmarks on Modal. Never run pytest locally.

  modal run modal_app.py::cpu_tests
  modal run modal_app.py::gpu_tests      # 1x H100, costs money: log in SPEND.md first
  modal run modal_app.py::bench_cpu --script bench.plots --args "overhead /results/k2a.csv /results/k2a"
  modal run modal_app.py::bench_gpu --script bench.linear_chain --args "--out /results/k2a.csv" --est-minutes 10

Outputs: bench scripts should write under `/results` (a persistent Modal Volume). Anything a script writes
under the repo's relative `results/` directory is also copied into the volume after the run. Fetch with:

  modal volume get loramux-results <remote-path> ./results
"""
import modal

app = modal.App("loramux")
results_vol = modal.Volume.from_name("loramux-results", create_if_missing=True)
RESULTS = "/results"

# One dependency list for every image, so test collection never fails on a missing import
# (tests import matplotlib, pandas, scipy and diffusers at module level).
_PY_DEPS = [
    "torch==2.8.0", "numpy", "safetensors", "pytest",
    "pandas", "matplotlib", "scipy",
    "diffusers==0.43.0", "transformers==5.23.0", "accelerate==1.16.0", "peft==0.22.0",
]
_src = lambda img: img.env({"PYTHONPATH": "/work"}).add_local_dir(
    ".", "/work", ignore=["results", ".git", "docs", "**/__pycache__"])
_cpu = modal.Image.debian_slim(python_version="3.11").pip_install(*_PY_DEPS)
_gpu = _cpu.pip_install("triton")
base = _src(_cpu)
gpu_img = _src(_gpu)
# Modal H100 SXM5 list rate, $0.001097/s (~$3.95/h), checked 2026-11-20 against third-party price trackers.
# GPU only: CPU and memory are billed on top. Re-check Modal's pricing page before a large run.
H100_USD_PER_HOUR = 3.95


def _run(cmd: list[str]) -> int:
    import subprocess
    return subprocess.run(cmd, check=False, cwd="/work").returncode


def _persist_results() -> list[str]:
    """Copy files written under /work/results into the volume, then commit it. Returns copied paths."""
    import os
    import shutil
    copied = []
    src = "/work/results"
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), src)
                dst = os.path.join(RESULTS, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(root, f), dst)
                copied.append(rel)
    results_vol.commit()
    return copied


@app.function(image=base, timeout=900, cpu=2)
def cpu_tests():
    return _run(["python", "-m", "pytest", "-q", "-m", "not gpu", "/work/tests"])


@app.function(image=gpu_img, gpu="H100!", timeout=1800)
def gpu_tests():
    return _run(["python", "-m", "pytest", "-q", "-m", "gpu", "/work/tests"])


@app.function(image=base, timeout=3600, cpu=2, volumes={RESULTS: results_vol})
def bench_cpu(script: str, args: str = "") -> dict:
    """Run `python -m <script> <args>`; outputs persist in the `loramux-results` volume."""
    import shlex
    rc = _run(["python", "-m", script, *shlex.split(args)])
    return {"returncode": rc, "copied_from_work_results": _persist_results()}


@app.function(image=gpu_img, gpu="H100!", timeout=3600, volumes={RESULTS: results_vol})
def bench_gpu(script: str, args: str, est_minutes: float) -> dict:
    """Run a bench script on 1x H100 after checking the spend caps against SPEND.md.

    The check reads the copy of SPEND.md inside the container, and the container cannot write back to it.
    After the run, log the actual cost yourself in your local SPEND.md (bench.spend.append_row) before the next run.
    Outputs persist in the `loramux-results` volume (write under /results, or the relative results/ directory).
    """
    import json
    import shlex

    from bench.common import collect_env
    from bench.spend import check_caps, estimate
    print(json.dumps(collect_env(), indent=2))
    est_usd = estimate("H100", est_minutes, H100_USD_PER_HOUR)
    check_caps("/work/SPEND.md", new_est=est_usd)  # raises SpendCapError before the script uses the GPU
    rc = _run(["python", "-m", script, *shlex.split(args)])
    return {"returncode": rc, "copied_from_work_results": _persist_results()}
