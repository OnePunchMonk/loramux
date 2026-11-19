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
