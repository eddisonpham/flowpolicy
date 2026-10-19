# 02. Environment setup

Status: Stage 1 draft. Research date and access date for every source: **2026-10-18**.
Legend: **[V]** verified against a source listed in section 10. **[S]** supported only by a secondary source (blog, third-party repo, forum). **[U]** unverified; must be checked on the owner's machine or in a later pass. Nothing here is recalled from memory without one of these tags.

Links: [00_overview](00_overview.md), [CHANGELOG](CHANGELOG.md).

---

## 1. Purpose and decisions made

Purpose: give a person or coding agent a runnable path from a bare machine to "all week-1 gates green", with pinned versions and known pitfalls.

Decisions made in this document (recorded properly as ADRs in Stage 2):

1. Install PyTorch from the **CUDA 13.0 (`cu130`)** index. Reason: it is PyTorch's stable CUDA variant and the one that covers Blackwell; 12.8 and 12.9 wheels no longer exist. [V]
2. Pin **PyTorch 2.14.1 and torchvision 0.29.1**. [V]
3. Train in Python **3.12**. Reason: ROS 2 Jazzy on Ubuntu 24.04 uses Python 3.12, so the training env and the ROS node can match; PyTorch 2.14 supports 3.10 to 3.15. [V for PyTorch and ROS; U for robosuite on 3.12]
4. Run the ROS 2 node **in Docker**, default distro **Jazzy**. Alternative: Lyrical. See section 6.
5. Use robomimic **only for dataset download and reading**, not for its training code. Reason: its published install docs and Dockerfile reference old Python versions. [V]

## 2. Machine fact sheet (owner fills in)

Measured 2026-10-18 on the owner's host. Commands are the Windows equivalents where the Linux command does not exist. The host is **not** Ubuntu; see `CHANGELOG.md` for the dated deviation.

| Fact | Command | Value |
|---|---|---|
| OS and version | `uname -a` (MINGW64) | Windows 11, build 26200, shell MINGW64 (Git Bash) |
| Native Linux or WSL2 | `uname -a` | Native Windows. No WSL2, no separate Ubuntu host (owner, 2026-10-18) |
| GPU model | `nvidia-smi -L` | NVIDIA GeForce RTX 5060 Laptop GPU |
| Total / free VRAM | `nvidia-smi --query-gpu=memory.total,memory.free --format=csv` | 8151 MiB total (7.93 GiB); free measured at 3428, 2640 and 6.88 GiB on three different checks — this number moves with whatever else is on the GPU, so measure it at run time |
| Driver version | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` | **617.42**, installed 2026-10-19, CUDA UMD 13.4 — meets the R580 minimum. Was 577.13 until 2026-10-19; that was the G1 cause |
| Compute capability | `torch.cuda.get_device_capability()` and `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` | 12.0 (sm_120), measured. `torch.cuda.get_arch_list()` contains `sm_120`, so the wheel covers this GPU [V 2026-10-19] |
| CPU cores | `nproc` | 24 |
| Python (project) | `.venv` built by uv | 3.12.13 |
| Python (host) | `python3 --version`, `python --version` | 3.14.5, 3.13.12 — neither is the project interpreter |
| Docker version | `docker --version` | **not installed** (G9a-G9c cannot run; deferred, M1.5) |
| uv version | `uv --version` | 0.11.30 |
| just version | `just --version` | 1.36.0 |
| torch installed | `uv pip show torch` | 2.14.1+cu130, resolved from `https://download.pytorch.org/whl/cu130` (Windows `win_amd64` wheel exists) |
| Triton | `uv pip show triton-windows` | `triton-windows==3.8.0.post29` (module `triton.__version__` = 3.8.0), added 2026-10-19 with a `sys_platform == 'win32'` marker. torch 2.14.1+cu130 does not pull it on Windows, so it is a separate pin |
| nvcc | `nvcc --version` | not installed (G6 cannot run; decision D6) |

The brief says "about 7 GB usable". Measured: 8151 MiB nominal, 3428 MiB free while the desktop is running. Plan all memory estimates against the free VRAM measured at run time, not the nominal figure, and re-measure with nothing else on the GPU.

## 3. Compatibility matrix

### 3.1 Tested-combination target (pin these; confirm with the gates in section 7)

| Component | Pinned target | Status | Notes |
|---|---|---|---|
| NVIDIA driver (Linux) | at least 580.65.06 for CUDA 13.0 GA; 580.95.05 for CUDA 13.0 Update 2 | [V] | Newer is fine. |
| PyTorch | 2.14.1 (released 2026-09-30) | [V] | 2.14.0 released 2026-09-02. |
| torchvision | 0.29.1 | [V] | |
| CUDA wheel variant | `cu130` | [V] | `cu132` is the newer option; see 3.2. |
| Python | 3.12 | [V] torch, ROS Jazzy; [U] robosuite | PyTorch 2.14 supports 3.10 to 3.15. |
| Triton | `triton-windows==3.8.0.post29`, Windows only (`sys_platform == 'win32'`) | [V] 2026-10-19 | Deviation from "whatever `torch==2.14.1` pins": on Windows the torch wheel ships no Triton at all, so the project pins a Windows build instead. Owner decision 2026-10-19. G3 and G5 pass with it. The marker keeps Linux CI on the torch-pinned Triton. [U] exact upstream-triton version this port tracks. |
| robosuite | 1.5.2 (PyPI release) | [S] | Needs `mujoco>=3.3.0` per a third-party repo; a third-party package lists `robosuite==1.5.2` with `mujoco==3.3.7` as its tested pair. |
| MuJoCo bindings | 3.3.x (start with 3.3.7) | [S] | Newer MuJoCo releases not checked. TBD. |
| robomimic | 0.5.0 (2025-06-20), installed from source for dataset tooling only | [V] | v0.4.0 added robosuite 1.5 support and moved datasets to Hugging Face. |
| gym-pusht | latest on PyPI (0.1.8 seen) | [S] | Adapted from Diffusion Policy; numerical equivalence with the original PushT env is **[U]**. Gate G7 plus a Stage 3 comparison decide. |
| ROS 2 | Jazzy Jalisco in Docker (default) | [V] | See section 6. |
| uv, Docker, NVIDIA Container Toolkit, Nsight | latest | [U] | Not researched in Stage 1. Verify during day one and record versions. |

### 3.2 CUDA variants in PyTorch 2.14

- Stable CUDA builds for 2.14 are 12.6, 13.0, and 13.2; CUDA 12.9 is dropped; the default wheel remains CUDA 13.0. [V]
- CUDA 12.8 was deprecated and removed starting with PyTorch 2.12. [V]
- CUDA 12.6 is kept as the "legacy" build for older GPU architectures (Maxwell, Pascal, Volta). [V] Do not use it for a Blackwell card.
- 2.14 is the last release that ships any CUDA 12.x wheels. [V]
- Blackwell support first appeared in the CUDA 12.8 wheels from PyTorch 2.7 (official PyTorch response on the forum). [S] That is history; it does not apply to the current wheels.
- After install, confirm the arch list includes your GPU: `python -c "import torch;print(torch.cuda.get_arch_list())"` must contain `sm_120` (or a PTX entry that covers it). If it does not, stop; you installed the wrong variant.

### 3.3 ROS 2 distro choices

| Distro | Ubuntu | Python | Support end | Status |
|---|---|---|---|---|
| Jazzy Jalisco | 24.04 | 3.12.3 | 2029-05-31 | [V] (support-end date from endoflife.date; Python from the Lyrical platform table, which lists Noble) |
| Kilted Kaiju | (24.04) | n/a | 2026-12-31 | [V] Do not start a new project on it. |
| Lyrical Luth (LTS, May 2026) | 26.04 Tier 1; 24.04 Tier 3 | 3.12 to 3.14 supported; 3.14.3 on 26.04 | May 2031 | [V] |

## 4. Known pitfalls (checked in the gates)

1. **Wrong CUDA wheel.** `cu126` installs, `torch.cuda.is_available()` may return true, but kernels for your arch are missing. Check the arch list (3.2).
2. **`ptxas` is no longer bundled in the cu13 PyTorch binary** (PyTorch 2.13 release blog). [V] Triton and Inductor both need a working assembler. Whether Triton's own bundled `ptxas` is enough on your machine is **[U]**; gate G5 tests it. If it fails, install the CUDA 13 toolkit (section 5.3) and set `TRITON_PTXAS_PATH` to its `ptxas`.
3. **No `nvcc` from pip PyTorch.** Gate G6 (CUDA C++ extension) needs the CUDA toolkit. Options in 5.3. The toolkit major version should match the wheel's CUDA (13.x).
4. **Installing a package that pins an old torch** (reported with xformers on Blackwell in a community guide). [S] Always install with the lockfile; run `uv pip list | grep -i torch` after any new dependency.
5. **Docker GPU passthrough** requires the NVIDIA Container Toolkit on the host. [U] Test with gate G9a.
6. **robomimic's docs are stale.** Its install page recommends Python 3.8 and its README says its Docker image uses Python 3.9. [V] Do not create the project env around those.
7. **MuJoCo rendering headless.** For image observations in robosuite you need an offscreen GL backend (EGL or OSMesa). Which one works on your box is **[U]**; gate G8 tests it. Try `MUJOCO_GL=egl` first.
8. **Version drift.** The lockfile (`uv.lock`) is the pin. The table above is the intent; the lockfile is the record. Commit both.

## 5. Step-by-step setup

### 5.1 Host prerequisites

```bash
# Driver check. Needs >= 580.65.06 for the cu130 wheel on Linux.
nvidia-smi
# If the driver is older, update via your distro's NVIDIA driver packages
# (method depends on distro; verify on your machine), then reboot.
```

### 5.2 Project and PyTorch (uv)

```bash
# Install uv if missing (see https://docs.astral.sh/uv/ ; version TBD)
mkdir flowpolicy && cd flowpolicy
uv init --python 3.12
# Install PyTorch from the cu130 index. Pins from the PyTorch 2.14 release thread.
uv add "torch==2.14.1" "torchvision==0.29.1" \
  --index "pytorch-cu130=https://download.pytorch.org/whl/cu130"
```

Notes:
- The index flag syntax above is **[U]**: check `uv add --help` for the version you have. The intent is: resolve `torch` and `torchvision` from `https://download.pytorch.org/whl/cu130` only, everything else from PyPI. The 2.14 RC thread shows the same index layout (`.../whl/cu130`, with `cu126` and `cu132` siblings). [V]
- If you prefer plain pip inside a uv venv: `uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`.
- To try CUDA 13.2 later, swap `cu130` for `cu132`. Treat that as an experiment, not the default.

Then the rest, added in groups so a failure is easy to localise:

```bash
uv add numpy scipy pandas matplotlib zarr hydra-core wandb
uv add --dev pytest ruff mypy pre-commit
# Simulators (see gates G7, G8 before committing to these):
uv add gym-pusht
uv add "robosuite==1.5.2" "mujoco==3.3.7"
```

robomimic, dataset tooling only:

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git third_party/robomimic
git -C third_party/robomimic checkout v0.5.0      # tag name is [U]; list tags to confirm
uv pip install --no-deps -e third_party/robomimic
# Then add only the dependencies you actually import (h5py is the likely one).
```

### 5.3 CUDA toolkit for `nvcc` (needed for gate G6 only)

Pick one; record the choice in the gate table.

- **A. Toolkit from NVIDIA's package repository** matching 13.x, installed system-wide. Most reliable for building extensions. Exact commands depend on distro: [U].
- **B. Toolkit inside the Docker image** (develop the optional CUDA stretch in a container only). Keeps the host clean.
- **C. Skip.** If neither works within about half a day, drop the CUDA C++ stretch (decision D6), as the brief already allows.

### 5.4 Docker

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi    # gate G9a
```

If this fails, install the NVIDIA Container Toolkit for your distro [U] and retry. Do not proceed to ROS 2 in Docker until this passes.

## 6. ROS 2 plan

Default (decision D7, to be ratified in ADR-007): **ROS 2 Jazzy in a Docker image**, with the policy node and the sim bridge each in a container (or one container with two nodes), communicating over ROS topics.

Why Jazzy by default:
- Python 3.12 matches the training environment, so the same checkpoint and the same torch wheel can load inside the container with no Python version split. [V for versions]
- Supported until 2029-05-31. [V]

Why consider Lyrical instead:
- It is the current LTS (released May 2026, supported to May 2031) and Tier 1 on Ubuntu 26.04. [V]
- Its Python on Ubuntu 26.04 is 3.14.3. [V] That diverges from the training env (3.12), and torch 2.14 supports 3.14 but robosuite and MuJoCo on 3.14 are **[U]**. If you take Lyrical, the policy node must not import robosuite; only the sim bridge does.

If the host is Ubuntu 24.04: Jazzy native is possible, but still prefer Docker so the node environment is reproducible and the CI/README story is one command.
If the host is Ubuntu 26.04: Lyrical native is Tier 1; Jazzy in Docker still works if the container image exists for it [U].

Fallback (brief section 8): plain Python fixed-rate loop with the same interfaces. Record the decision in an ADR if you use it.

## 7. Smoke tests (week 1 gates)

Put these in `scripts/gates/` and run each from the repo root with `uv run python scripts/gates/<file>.py`. Each prints `PASS` or `FAIL` plus the facts needed to debug. They contain no measured numbers; results go in section 8.

**G1: GPU and arch** (`g1_gpu.py`)

```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
free, total = torch.cuda.mem_get_info()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", name, "capability", cap)
print("arch_list", arch)
print("vram_total_GiB", total / 2**30, "free_GiB", free / 2**30)
sm = f"sm_{cap[0]}{cap[1]}"
assert any(a.startswith(sm) or a.startswith("compute_") for a in arch), f"no kernels for {sm}"
x = torch.randn(1024, 1024, device="cuda")
assert torch.isfinite((x @ x).sum()).item()
print("PASS")
```

**G2: bf16 autocast and SDPA** (`g2_bf16_sdpa.py`)

```python
import torch, torch.nn.functional as F
q = torch.randn(2, 8, 64, 32, device="cuda", dtype=torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = F.scaled_dot_product_attention(q, q, q)
assert out.shape == q.shape and torch.isfinite(out.float()).all()
print("PASS")
```

**G3: `torch.compile`, all three modes** (`g3_compile.py`)

```python
import torch
m = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)).cuda().eval()
x = torch.randn(1, 64, device="cuda")
ref = m(x)
for mode in ["default", "reduce-overhead", "max-autotune"]:
    c = torch.compile(m, mode=mode)
    with torch.no_grad():
        for _ in range(3):
            y = c(x)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
    print(mode, "PASS")
```

**G4: CUDA graph capture of a small loop** (`g4_cudagraph.py`)

```python
import torch
m = torch.nn.Linear(64, 64).cuda().eval()
static_x = torch.randn(1, 64, device="cuda")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.no_grad():
    for _ in range(3):
        m(static_x)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad():
    static_y = m(static_x)
static_x.copy_(torch.randn(1, 64, device="cuda"))
g.replay()
torch.testing.assert_close(static_y, m(static_x), rtol=1e-5, atol=1e-5)
print("PASS")
```

**G5: Triton hello world** (`g5_triton.py`)

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

n = 4096
x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
o = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
torch.testing.assert_close(o, x + y)
print("triton", triton.__version__, "PASS")
```

If G5 fails with an assembler error, see pitfall 2.

**G6: minimal CUDA extension** (`g6_cuda_ext.py`). Requires `nvcc` (5.3).

```python
import torch
from torch.utils.cpp_extension import load_inline
cuda_src = r"""
__global__ void scale_k(const float* x, float* y, float a, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i];
}
torch::Tensor scale(torch::Tensor x, double a){
  auto y = torch::empty_like(x);
  int n = x.numel();
  scale_k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), (float)a, n);
  return y;
}
"""
cpp_src = "torch::Tensor scale(torch::Tensor x, double a);"
ext = load_inline("g6_ext", cpp_src, cuda_src, functions=["scale"], verbose=True)
x = torch.randn(1000, device="cuda")
torch.testing.assert_close(ext.scale(x, 3.0), 3.0 * x)
print("PASS")
```

**G7: PushT closed loop, random policy** (`g7_pusht.py`)

```python
import gymnasium as gym, gym_pusht   # noqa: F401
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, info = env.reset(seed=0)
for _ in range(200):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
print("PASS")
```

The `obs_type` string above is **[U]**; check the package README for the accepted values. Also write down which PushT implementation you chose and why (gym-pusht vs the Diffusion Policy repo's env); equivalence is tested in Stage 3.

**G8: robosuite Lift, random policy, with and without images** (`g8_robosuite.py`)

```python
import os, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite as suite
for use_cam in (False, True):
    env = suite.make("Lift", robots="Panda", has_renderer=False,
                     has_offscreen_renderer=use_cam, use_camera_obs=use_cam,
                     camera_names="agentview" if use_cam else None,
                     camera_heights=84, camera_widths=84, control_freq=20)
    env.reset()
    low, high = env.action_spec
    for _ in range(100):
        env.step(np.random.uniform(low, high))
    env.close()
    print("camera_obs", use_cam, "PASS")
```

Then, separately, download one robomimic dataset file (Lift, proficient-human, low-dim) from the Hugging Face location robomimic v0.4 moved to (exact repo name **[U]**; follow the robomimic README), open it with `h5py`, and print its keys. Record the license from the dataset page.

**G9a: Docker GPU passthrough** (see 5.4).
**G9b: ROS 2 hello pub/sub in Docker**

```bash
docker run --rm osrf/ros:jazzy-desktop bash -lc \
 "source /opt/ros/jazzy/setup.bash && ros2 doctor --report | head -n 20"
# Image tag exists: [U]. If the pull fails, check the tag list on Docker Hub.
```

Then a two-container test: one container runs `ros2 run demo_nodes_cpp talker`, the other `ros2 run demo_nodes_cpp listener`, on the same Docker network. DDS discovery between containers on separate networks is a common failure; use `--network host` for the smoke test.

**G9c: torch inside the ROS container.** Build the container FROM a ROS Jazzy base, install the same `torch==2.14.1+cu130` wheel into a venv, and run G1 there with `--gpus all`. This proves the node can load a checkpoint.

## 8. Gate results table (owner and coding agent fill in)

| Gate | What it proves | Result | Date | Notes |
|---|---|---|---|---|
| G1 | GPU visible, correct arch | PASS | 2026-10-19 | `torch 2.14.1+cu130`, cuda 13.0, device `NVIDIA GeForce RTX 5060 Laptop GPU`, capability (12, 0), `arch_list` contains `sm_120`, VRAM 7.93 GiB total / 6.88 GiB free. The 2026-10-18 FAIL was the driver (577.13); 617.42 fixed it. |
| G2 | bf16 autocast and SDPA | PASS | 2026-10-19 | `out_dtype torch.bfloat16 shape (2, 8, 64, 32)`. |
| G3 | `torch.compile` (3 modes) | PASS | 2026-10-19 | All three modes compile and run: `default` 4.09 s / 3.04 s, `reduce-overhead` 1.16 s / 0.14 s, `max-autotune` 2.11 s / 0.10 s on two runs (first run pays the compile cost; not a benchmark). Needed `triton-windows==3.8.0.post29`; see §3.1. |
| G4 | CUDA graph capture | PASS | 2026-10-19 | Capture and replay matched the eager result at rtol/atol 1e-5. |
| G5 | Triton kernel runs | PASS | 2026-10-19 | `triton 3.8.0 PASS`: kernel compiled, launched and matched `x + y` via `assert_close`. This also clears risk R2 for this host — the assembler worked, so no `TRITON_PTXAS_PATH` workaround is needed. |
| G6 | CUDA C++ extension builds | NOT RUN | 2026-10-19 | `nvcc` missing; decision D6 stays closed. `scripts/gates/g6_cuda_extension.py` reports NOT RUN. |
| G7 | PushT closed loop | PASS | 2026-10-19 | `scripts/gates/g7_pusht_random.py`: 3 random episodes, observation `(96,96,3)` uint8 and state `(2,)`, 300 steps each. Success is not expected for a random policy; this proves reset/step/render/JSONL. Rows in `results/gates/g7.jsonl`. |
| G8 | robosuite Lift closed loop, plus dataset read | NOT RUN | 2026-10-19 | robosuite not installed; fallback D1 = PushT only (Windows host, no EGL path attempted). |
| G9a | Docker GPU passthrough | NOT RUN | 2026-10-19 | `docker` not installed; fallback D7 = plain Python loop. |
| G9b | ROS 2 pub/sub | NOT RUN | 2026-10-19 | Depends on G9a. |
| G9c | torch in ROS container | NOT RUN | 2026-10-19 | Depends on G9a. |

History: the 2026-10-18 run was five FAILs with one root cause, the driver (577.13). Updating to 617.42 turned G1, G2 and G4 green with the scripts unedited, and left G3 and G5 red for a second reason: no `triton` on Windows. Adding `triton-windows==3.8.0.post29` as a Windows-only dependency (owner decision, §3.1) turned those two green. G7 was added and passed on 2026-10-19 once the PushT data and `gym-pusht` were installed. G6, G8 and G9a-G9c remain NOT RUN with the fallbacks above; none is treated as a pass.

**Fallback mapping (from the brief):** G1-G5 red means stop and fix before any model or data work (risk R1, no fallback); G8 red means drop robomimic and run PushT only; G9a, G9b, or G9c red means defer the ROS 2 decision and plan for the plain Python control loop; G6 red means drop the CUDA stretch; G3, G4, or G5 red must be fixed before any other work.

## 9. Acceptance criteria

A later coding agent can check this document is done when:

- [ ] Section 2 has real values from the owner's machine, no TBD left.
- [ ] Section 3.1 has the Triton, MuJoCo, uv, Docker, and Nsight versions filled in from `pip show` or `--version`, and `uv.lock` is committed.
- [ ] Every row in section 8 is PASS, or has a recorded fallback decision.
- [ ] G1 output shows `torch 2.14.1`, CUDA 13.0, and an arch list that covers the GPU.
- [ ] The "unverified" tags in section 3.1 are each either confirmed (tag changed to [V] with a source) or turned into an explicit risk in `10_risk_register.md`.

## 10. Verified references (access date 2026-10-18 for all)

PyTorch
- PyTorch 2.14.1 General Availability (2026-09-30; torchvision 0.29.1): https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455
- Same thread lists 2.14.0 GA on 2026-09-02 and 2.13.0 on 2026-07-08: https://dev-discuss.pytorch.org/t/pytorch-2-13-0-general-availability/3412
- 2.14 feature call: CUDA 12.6, 13.0, 13.2; 12.9 dropped; default wheel CUDA 13.0; Python 3.10 to 3.15: https://dev-discuss.pytorch.org/t/reminder-call-for-features-pytorch-2-14/3423
- 2.14 RC1 install index layout (`.../whl/cu126`, `cu130`, `cu132`): https://dev-discuss.pytorch.org/t/pytorch-2-14-rc1-produced-for-pytorch-torchvision/3422
- CUDA 13.2 introduced, CUDA 12.8 deprecated (2.12): https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337
- Release/packaging notices (2.14 last release with Python 3.10 wheels and any CUDA 12.x wheels): https://dev-discuss.pytorch.org/c/release-packaging/25.md
- PyTorch 2.13 release blog (ptxas no longer bundled in the cu13 binary; CUDA 12.8/12.9 builds removed): https://pytorch.org/blog/pytorch-2-13-release-blog/
- Forum: Blackwell support since 2.7 with CUDA 12.8 wheels [S]: https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621

NVIDIA
- CUDA 13.0 Update 2 release notes (driver minimums: 13.0 GA at least 580.65.06; Update 1 at least 580.82.07; Update 2 at least 580.95.05): https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html.md

ROS 2
- Lyrical Luth release page (Python 3.12 to 3.14): https://docs.ros.org/en/lyrical/Releases/Release-Lyrical-Luth.html
- Lyrical supported platforms (Ubuntu 26.04 Tier 1; Noble Tier 3; Python 3.14.3 on 26.04, 3.12.3 on 24.04): https://docs.ros.org/en/lyrical/Releases/lyrical/supported-platforms.html
- Lyrical timeline (end of life May 2031): https://docs.ros.org/en/jazzy/Releases/lyrical/release-timeline.html
- Support-end dates (Jazzy 2029-05-31, Kilted 2026-12-31): https://endoflife.date/ros-2

Simulators and datasets
- robomimic v0.5.0 release notes: https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.5.0
- robomimic README mirror (v0.4.0: robosuite v1.5 support, datasets on Hugging Face; Docker uses Python 3.9): https://gitee.com/fandes/robomimic
- robomimic install docs (recommends Python 3.8): https://robomimic.github.io/docs/introduction/installation.html
- robosuite 1.5.2 needing `mujoco>=3.3.0` [S]: https://github.com/TensorAuto/OpenTau/pull/386
- robosuite 1.5.2 with mujoco 3.3.7 as a third-party tested pair [S]: https://pypi.org/project/nomad-harness/
- gym-pusht (adapted from Diffusion Policy; version 0.1.8 seen) [S]: https://pypi.org/project/gym-pusht/

Not verified in Stage 1: exact Triton pin for torch 2.14.1; latest MuJoCo release; uv index syntax; NVIDIA Container Toolkit install steps; Nsight versions; the Hugging Face repo names for robomimic datasets; robosuite on Python 3.12; the `osrf/ros:jazzy-desktop` tag.

## 11. Next actions

- [ ] Owner: fill sections 2 and 8; paste failures verbatim.
- [ ] Agent: resolve every item under "Not verified in Stage 1" above, one at a time, upgrading tags.
- [ ] Agent: turn section 4 pitfalls into entries in `10_risk_register.md`.
- [ ] Agent (Stage 2): ADR-007 (deployment layer) and ADR-001/002 (benchmarks, robomimic scope) from the evidence here.
