# SD Forge + DirectML on AMD (RX 6600 XT)

Turn‑key setup to run **Stable Diffusion Forge** on Windows with **DirectML** on an **AMD Radeon RX 6600 XT (8 GB)**. Includes four launch profiles (Low / Medium / High / XL), memory tuning, and attention strategies that avoid the classic *Could not allocate tensor… not enough GPU video memory* crashes.

> ✅ Tested with Python **3.10.6**, Forge **f2.0.1v1.10.1‑previous‑669**, PyTorch **2.4.1+cpu** + **torch‑directml**, Windows 10/11, VRAM 8 GB.

---

## What’s in here

* **Four launchers** (`run_low.bat`, `run_medium.bat`, `run_high.bat`, `run_xl.bat`) tuned for 8 GB AMD cards.
* **DirectML device selection** (use device 0 by default; configurable).
* **Stable attention modes**:

  * *Split attention* (very VRAM‑friendly; safer on DirectML).
  * *Sub‑quadratic attention* (aggressive; use when it works on your setup).
* **VAE on CPU** to save VRAM (`--vae-in-cpu`).
* **Low/No‑VRAM strategies** (`--always-low-vram`, optional `--always-offload-from-vram`).
* **Softmax budget** via `inference_memory` (matrix compute pool) set around **2 GB** for 8 GB cards.

---

## Folder layout

Place the BAT files **at the root of your Forge checkout** (same level as `launch.py`). Example:

```
D:\AI\sd-forge-RX6600XT\
├─ launch.py
├─ webui-user.bat
├─ venv\
├─ models\
├─ run_low.bat
├─ run_medium.bat
├─ run_high.bat
└─ run_xl.bat
```

> If you prefer a subfolder (e.g., `scripts\`), adjust the `pushd`/`PYTHON` lines to point back to the repo root.

---

## Requirements

* **Python 3.10.6** (exact minor matters for Forge).
* A cloned **sd-forge** repo.
* **torch-directml** (the BATs check and install automatically if missing).
* An AMD GPU with **8 GB VRAM** (tested on **RX 6600 XT**).

---

## Launch profiles

Each BAT prints its profile banner and the resolved arguments before launching.

### 1) Low (safest)

* Best chance to avoid OOM.
* Forces split attention and optional VRAM offload.
* `inference_memory` ≈ **1792 MB**.

**Flags (summary):**

```
--directml 0 --skip-install --skip-torch-cuda-test --precision full --no-half \
--always-low-vram --always-offload-from-vram --vae-in-cpu --attention-split
```

### 2) Medium (recommended)

* Split attention, no forced offload.
* `inference_memory` ≈ **2048 MB**.

**Flags (summary):**

```
--directml 0 --skip-install --skip-torch-cuda-test --precision full --no-half \
--always-low-vram --vae-in-cpu --attention-split
```

### 3) High (more aggressive)

* Tries **sub‑quadratic attention** (faster when it fits), still Low‑VRAM + VAE on CPU.
* Keep an eye on OOM messages; fall back to Medium if needed.

**Flags (summary):**

```
--directml 0 --skip-install --skip-torch-cuda-test --precision full --no-half \
--always-low-vram --vae-in-cpu --opt-sub-quad-attention
```

### 4) XL (fast path on our setup)

* Tuned set that ran quickest in our tests at 512×640 / 20 steps.
* Use when everything else is stable and you want throughput.

**Flags (summary):**

```
--directml 0 --skip-install --skip-torch-cuda-test --precision full --no-half \
--always-low-vram --vae-in-cpu --opt-sub-quad-attention
```

> **Note:** Some Forge builds expose the flag as `--opt-sub-quad-attention`, others enable sub‑quad automatically when available. If you see “unrecognized argument”, switch to the Medium profile (split attention) or remove the flag.

---

## How memory is managed

* **Weights vs compute budget**: Forge logs something like:

  * *“You will use 87.50% GPU memory to load weights, and 12.50% for matrix computation.”*

* We explicitly set a compute pool via the env var `inference_memory` (**MB**). Our launchers aim for **\~2 GB** (or \~1.75 GB for Low). If you still see OOM around `torch.baddbmm` / softmax:

  1. Lower resolution (e.g., from **512×640 → 512×512**).
  2. Switch to **Medium/Low** profile.
  3. Reduce `inference_memory` to **1536** or **1280** MB.
  4. Avoid **Hires. fix** for now on DirectML (costly).

* **VAE on CPU**: saves a meaningful chunk of VRAM on 8 GB cards and is stable with DirectML.

* **xFormers**: not required on DirectML; those warnings can be ignored.

---

## Model used in testing

* **Checkpoint**: `realisticVisionV60B1_v60B1VAE.safetensors`
* Resolution baseline: **512×640**, **20 steps**, **CFG 7**, sampler **Euler a** or **DPM++ 2M**.

### Sample prompts

**Positive**

```
a medium dog sitting on a mat in a leafy park, green collar, photorealistic, 85mm, high detail
```

**Negative**

```
text, watermark, extra limbs, deformed, blurry, lowres
```

**Samurai example**

```
A lone samurai in traditional armor, standing before a rising sun flag motif,
cinematic lighting, misty Japanese landscape, long katana, intricate details,
sharp focus, 85mm, photorealistic, volumetric light, high dynamic range
```

---

## Quick start

1. Copy the four `run_*.bat` files into your Forge root.
2. Double‑click the profile you want (Low/Medium/High/XL).
3. Open the Gradio URL shown (usually `http://127.0.0.1:7860`).
4. Start with **512×640**, **20 steps**, **CFG 7**, **batch size 1**.

> The launchers will auto‑install **torch‑directml** if missing.

---

## Troubleshooting

### "Could not allocate tensor … 1,342,177,280 bytes"

This is a softmax/attention burst. Try:

* Switch to **Medium** (split attention) or **Low** (adds offload).
* Reduce resolution or steps; keep batch size at **1**.
* Lower `inference_memory` (e.g., 2048 → 1792 → 1536 MB).

### "xformers not found"

Safe to ignore on **DirectML**—we don’t use xFormers in these profiles.

### Memory monitor disabled / CUDA warnings

Expected with **PyTorch+CPU + DirectML bridge**. Not a problem.

### Where to change the DirectML device

Edit the BAT: set `DML_DEVICE=0` (or leave empty for default). Use `1` if you have multiple GPUs.

---

## Bench notes (indicative only)

Observed on RX 6600 XT, 512×640, 20 steps (single image):

| Profile | Attention | Offload | Softmax pool | Outcome (logs)                                               |
| ------- | --------- | ------: | -----------: | ------------------------------------------------------------ |
| Low     | Split     |     Yes |    \~1792 MB | Stable, slowest, few OOM retries then completes              |
| Medium  | Split     |      No |    \~2048 MB | Stable in most runs; occasional softmax retries              |
| High    | Sub‑quad  |      No |    \~2048 MB | Can be fastest **when** it fits; otherwise OOMs → use Medium |
| XL      | Sub‑quad  |      No |    \~2048 MB | Fast path on our setup; use after verifying stability        |

> Your times will vary by driver, background apps, and model.

---

## License & model rights

* **Repo code**: recommended to ship under **MIT License** (see `LICENSE` in the repo).
* **Model files** (e.g., `realisticVisionV60B1_v60B1VAE.safetensors`) are **not included**. Respect the model’s original license and distribution terms.

Template `LICENSE` (MIT):

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Credits

* **Stable Diffusion Forge** and its authors.
* Everyone hacking on DirectML + AMD workflows.

If you tweak the launchers for other AMD cards (6700/6800/7800, etc.), PRs welcome! ✨
