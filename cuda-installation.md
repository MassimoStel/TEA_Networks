# CUDA + CuPy Installation Guide

## Prerequisites

Before starting, confirm you have:

- An NVIDIA GPU (RTX 2000 Ada or similar)
- NVIDIA driver version 580.92 or later already installed
- A Python virtual environment (e.g. `teanets-env2`)
- PowerShell or Command Prompt with administrator rights

---

## Step 1 — Download CUDA Toolkit 12.6

**Download URL:** <https://developer.nvidia.com/cuda-12-6-0-download-archive>

Select the following options on the download page:

| Option | Value |
|---|---|
| Operating System | Windows |
| Architecture | x86_64 |
| Version | 11 (or 10, whichever matches your Windows) |
| Installer Type | exe (local) |

> **Note:** Although your driver reports CUDA 13.0 support, CuPy does not yet support CUDA 13. CUDA 12.6 is fully compatible with your driver and hardware.

Run the downloaded installer and choose **Express Installation**. The installer will set up CUDA and update your `PATH` automatically.

---

## Step 2 — Install CuPy in Your Virtual Environment

Open PowerShell and run the following commands:

```powershell
cd d:\UniTrento\TeaNets\TEA_Networks
.\teanets-env2\Scripts\activate
pip install cupy-cuda12x
```

> **Note:** Always activate your virtual environment before installing packages, so they are available to your Jupyter notebooks.

---

## Step 3 — Verify the Installation

### 3a. Verify CUDA Toolkit

Open a **new** PowerShell window and run:

```powershell
nvcc --version
```

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
release 12.6, V12.6.20
```

### 3b. Verify CuPy in Jupyter

In your Jupyter notebook, run the following cell:

```python
import cupy as cp

# Should print 12060 (or similar for CUDA 12.6)
print(cp.cuda.runtime.runtimeGetVersion())

# Should print array([1, 2, 3]) — running on GPU
a = cp.array([1, 2, 3])
print(a)
```

> **Note:** The CuPy warning *"CUDA path could not be detected"* should no longer appear after completing Steps 1–2.

---

## Step 4 — Set `CUDA_PATH` Environment Variable (if needed)

The CUDA installer usually sets this automatically. If the warning persists, set it manually:

`Win + R` → `sysdm.cpl` → **Advanced** → **Environment Variables** → **System Variables** → **New**

| Field | Value |
|---|---|
| Variable name | `CUDA_PATH` |
| Variable value | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` |

Also add the following to the `Path` variable:

- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp`

After setting environment variables, **restart PowerShell and Jupyter Kernel** before testing again.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `nvcc` not found after install | Close and reopen PowerShell. If still missing, add `CUDA\v12.6\bin` to `PATH` manually (see Step 4). |
| CuPy warning still appears | Confirm `CUDA_PATH` is set correctly and restart the Jupyter kernel (*Kernel → Restart*). |
| `pip install cupy-cuda12x` fails | Make sure the virtual environment is activated first (`.\teanets-env2\Scripts\activate`). |
| Wrong CUDA version installed | Your driver supports up to CUDA 13, but always use `cupy-cuda12x` until CuPy adds CUDA 13 support. |
