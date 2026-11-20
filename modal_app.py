"""Run tests and benchmarks on Modal. Never run pytest locally.

  modal run modal_app.py::cpu_tests
  modal run modal_app.py::gpu_tests      # 1x H100, costs money: log in SPEND.md first
"""
import modal

app = modal.App("loramux")
_src = lambda img: img.env({"PYTHONPATH": "/work"}).add_local_dir(
    ".", "/work", ignore=["results", ".git", "docs", "**/__pycache__"])
_cpu = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "numpy", "safetensors", "pytest")
base = _src(_cpu)
gpu_img = _src(_cpu.pip_install("triton"))
# Bench images add the bench extra. add_local_dir must stay last, so these are built from _cpu directly.
bench_cpu_img = _src(_cpu.pip_install("pandas", "matplotlib", "scipy"))
bench_gpu_img = _src(_cpu.pip_install("triton", "pandas", "matplotlib", "scipy"))
# UNVERIFIED: recalled Modal on-demand H100 rate. Must be verified against Modal's current price list before any GPU run.
H100_USD_PER_HOUR = 3.95


@app.function(image=base, timeout=900, cpu=2)
def cpu_tests():
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-m", "not gpu", "/work/tests"], cwd="/work")
    return r.returncode


@app.function(image=gpu_img, gpu="H100!", timeout=1800)
def gpu_tests():
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-m", "gpu", "/work/tests"], cwd="/work")
    return r.returncode


@app.function(image=bench_cpu_img, timeout=3600, cpu=2)
def bench_cpu(script: str, args: str = "") -> int:
    import shlex
    import subprocess
    r = subprocess.run(["python", "-m", script, *shlex.split(args)], check=False, cwd="/work")
    return r.returncode


@app.function(image=bench_gpu_img, gpu="H100!", timeout=3600)
def bench_gpu(script: str, args: str, est_minutes: float) -> int:
    """Run a bench script on 1x H100 after checking the spend caps against SPEND.md.

    The check reads the copy of SPEND.md inside the container, and the container cannot write back to it.
    After the run, log the actual cost yourself in your local SPEND.md (bench.spend.append_row) before the next run.
    """
    import json
    import shlex
    import subprocess

    from bench.common import collect_env
    from bench.spend import check_caps, estimate
    print(json.dumps(collect_env(), indent=2))
    est_usd = estimate("H100", est_minutes, H100_USD_PER_HOUR)
    check_caps("/work/SPEND.md", new_est=est_usd)  # raises SpendCapError before any GPU time is used
    r = subprocess.run(["python", "-m", script, *shlex.split(args)], check=False, cwd="/work")
    return r.returncode
