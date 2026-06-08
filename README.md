# VisionForge 🔮

Local image recognition + Stable Diffusion 1.5 generation pipeline.
No cloud API needed — runs 100% on your machine.

---

## What it does

1. **Analyze** any image using BLIP (vision-language model)  
   → Generates captions: detailed, artistic, unconditional  
   → Extracts keywords automatically

2. **Generate** new images via Stable Diffusion 1.5  
   → Builds an optimized prompt from the analysis  
   → Saves output PNGs to `visionforge_output/`

3. **Export training data**  
   → Every run appends a JSON entry to `training_data.jsonl`  
   → Use this dataset to fine-tune your own model

---

## Setup

### 1. Install Python 3.9+
https://python.org

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install PyTorch
**CPU only (slower but works everywhere):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (much faster):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install other dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Interactive mode (recommended)
```bash
python visionforge.py
```

Commands inside the REPL:
```
analyze  <path>      Analyze an image file
generate [prompt]    Generate from last analysis
prompt   <text>      Set a custom generation prompt
steps    <n>         Set inference steps (default 30)
cfg      <n>         Set CFG scale (default 7.5)
seed     <n>         Set random seed for reproducibility
keywords             Show extracted keywords
history              Show training data log
help                 Show all commands
quit                 Exit
```

### Batch mode (one-shot)
```bash
# Analyze + generate from an image
python visionforge.py --image photo.jpg

# Add a custom style prompt
python visionforge.py --image photo.jpg --prompt "oil painting, Van Gogh style"

# Higher quality (more steps)
python visionforge.py --image photo.jpg --steps 50 --cfg 8.0

# Custom output folder
python visionforge.py --image photo.jpg --output-dir my_outputs
```

---

## Models used

| Model | Purpose | Size |
|-------|---------|------|
| `Salesforce/blip-image-captioning-large` | Image → text captions | ~900 MB |
| `runwayml/stable-diffusion-v1-5` | Text → image generation | ~4 GB |

Models download automatically on first run and are cached in `~/.cache/huggingface/`.

---

## Training data format

Each generation appends to `visionforge_output/training_data.jsonl`:

```json
{
  "id": "entry_1718400000",
  "timestamp": "2024-06-15T10:00:00",
  "source_image": "photo.jpg",
  "captions": {
    "caption": "a photo of a mountain lake at sunset",
    "detailed": "this image shows a serene alpine lake...",
    "artistic": "an artistic depiction of golden hour...",
    "unconditional": "a lake surrounded by pine trees"
  },
  "training_pair": {
    "input_prompt": "a mountain lake at sunset, highly detailed, 8k...",
    "negative_prompt": "blurry, low quality...",
    "output_image": "visionforge_output/generated_20240615_100000.png"
  }
}
```

Use this JSONL file to fine-tune models with tools like:
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — SD fine-tuning
- [huggingface/diffusers training scripts](https://github.com/huggingface/diffusers/tree/main/examples)

---

## Hardware recommendations

| Setup | Speed |
|-------|-------|
| CPU only | ~3–5 min per image |
| NVIDIA GPU (8GB+) | ~15–30 sec per image |
| NVIDIA GPU + xformers | ~8–15 sec per image |

For GPU + xformers: `pip install xformers` then uncomment the line in `requirements.txt`.

---

## Tips

- Start with `steps 20` for quick previews, `steps 50` for final quality
- CFG scale: lower (5–6) = more creative, higher (8–10) = follows prompt strictly
- Use `seed <number>` to reproduce the same image with different prompts
- Analyze multiple images to build a richer training dataset
