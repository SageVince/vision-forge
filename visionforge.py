#!/usr/bin/env python3
"""
VisionForge — Local Image Recognizer + Generator
Uses BLIP for image recognition and Stable Diffusion 1.5 for generation.
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime


# ── Lazy imports (shown after dependency check) ──────────────────────────────

def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "diffusers", "PIL", "accelerate"]:
        try:
            __import__("PIL" if pkg == "PIL" else pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[!] Missing packages. Install with:\n")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("    pip install transformers diffusers accelerate Pillow\n")
        sys.exit(1)


# ── ANSI colors ───────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    PURPLE = "\033[35m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

def banner():
    print(f"""
{C.PURPLE}{C.BOLD}
  ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
  ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
  ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
  ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
   ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
    ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
{C.CYAN}        F  O  R  G  E{C.RESET}
{C.GRAY}  Image Recognizer + SD 1.5 Generator{C.RESET}
""")

def hdr(msg):
    print(f"\n{C.PURPLE}{C.BOLD}{'─'*50}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {msg}{C.RESET}")
    print(f"{C.PURPLE}{C.BOLD}{'─'*50}{C.RESET}")

def info(msg):  print(f"{C.GRAY}  → {msg}{C.RESET}")
def ok(msg):    print(f"{C.GREEN}  ✓ {msg}{C.RESET}")
def warn(msg):  print(f"{C.YELLOW}  ⚠ {msg}{C.RESET}")
def err(msg):   print(f"{C.RED}  ✗ {msg}{C.RESET}")
def label(k,v): print(f"  {C.WHITE}{k:<18}{C.RESET}{C.CYAN}{v}{C.RESET}")


# ── Image Recognition (BLIP) ──────────────────────────────────────────────────

def load_recognizer():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    hdr("Loading Recognition Model")
    info("Model: Salesforce/blip-image-captioning-large")
    info("Downloading on first run (~900 MB), cached after…")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device.upper()}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    ok("Recognition model ready")
    return processor, model, device


def recognize_image(image_path: str, processor, model, device):
    from PIL import Image
    import torch

    hdr("Analyzing Image")
    info(f"File: {image_path}")

    img = Image.open(image_path).convert("RGB")
    label("Size:", f"{img.width} × {img.height} px")

    # Generate multiple captions with different prompts
    results = {}
    prompts = [
        ("caption",    "a photo of"),
        ("detailed",   "this image shows"),
        ("artistic",   "an artistic depiction of"),
    ]

    for key, prompt_text in prompts:
        inputs = processor(img, prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=5,
                repetition_penalty=1.3
            )
        results[key] = processor.decode(out[0], skip_special_tokens=True)

    # Also run unconditional caption
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, num_beams=5)
    results["unconditional"] = processor.decode(out[0], skip_special_tokens=True)

    ok("Analysis complete")
    return results, img.size


def extract_keywords(captions: dict) -> list[str]:
    """Extract meaningful keywords from captions for prompt building."""
    import re
    stopwords = {
        "a","an","the","this","is","of","in","on","at","to","and","or",
        "with","photo","image","shows","depiction","artistic","there","has",
        "are","it","its","that","for","from","by","as","be","was","were"
    }
    words = set()
    for caption in captions.values():
        tokens = re.findall(r"[a-z]+", caption.lower())
        for t in tokens:
            if t not in stopwords and len(t) > 3:
                words.add(t)
    return sorted(words)[:15]


# ── Prompt Engineering ────────────────────────────────────────────────────────

def build_generation_prompt(captions: dict, user_prompt: str = "") -> dict:
    """
    Build an optimized SD 1.5 prompt from captions.
    Returns positive + negative prompts.
    """
    base = captions.get("detailed") or captions.get("caption", "")

    if user_prompt:
        positive = f"{user_prompt}, {base}, highly detailed, 8k uhd, photorealistic, sharp focus, studio lighting"
    else:
        positive = f"{base}, highly detailed, 8k uhd, photorealistic, sharp focus, professional photography, masterpiece"

    negative = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "watermark, signature, text, noise, oversaturated, low resolution"
    )
    return {"positive": positive, "negative": negative}


# ── Image Generation (SD 1.5) ─────────────────────────────────────────────────

def load_generator():
    from diffusers import StableDiffusionPipeline
    import torch

    hdr("Loading Generation Model")
    info("Model: runwayml/stable-diffusion-v1-5")
    info("Downloading on first run (~4 GB), cached after…")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    if device == "cpu":
        warn("No GPU detected — generation will be slow (2–5 min per image)")
        warn("For faster results, run on a CUDA-capable GPU")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,          # disable for speed; re-enable if needed
        requires_safety_checker=False
    )
    pipe = pipe.to(device)

    # Memory optimizations
    if device == "cpu":
        pipe.enable_attention_slicing()
    else:
        pipe.enable_xformers_memory_efficient_attention()

    ok("Generation model ready")
    return pipe, device


def generate_image(pipe, prompts: dict, output_path: str, steps: int = 30, cfg: float = 7.5, seed: int = None):
    import torch

    hdr("Generating Image")
    info(f"Steps: {steps}  |  CFG scale: {cfg}")
    info(f"Positive: {prompts['positive'][:80]}…")
    info(f"Negative: {prompts['negative'][:60]}…")

    generator = torch.Generator().manual_seed(seed) if seed else None

    result = pipe(
        prompt=prompts["positive"],
        negative_prompt=prompts["negative"],
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=512,
        height=512,
        generator=generator
    )

    image = result.images[0]
    image.save(output_path)
    ok(f"Saved → {output_path}")
    return output_path


# ── Training Data Export ──────────────────────────────────────────────────────

def save_training_entry(captions, prompts, image_path, output_dir):
    entry = {
        "id": f"entry_{int(datetime.now().timestamp())}",
        "timestamp": datetime.now().isoformat(),
        "source_image": image_path,
        "captions": captions,
        "training_pair": {
            "input_prompt":       prompts["positive"],
            "negative_prompt":    prompts["negative"],
            "output_image":       str(output_dir / "generated.png"),
        }
    }
    log_path = output_dir / "training_data.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    ok(f"Training entry appended → {log_path}")
    return entry


# ── Interactive REPL ──────────────────────────────────────────────────────────

def interactive_mode(recognizer_args, generator_args):
    """Full interactive pipeline."""
    banner()
    print(f"  {C.GRAY}Type 'help' for commands, 'quit' to exit{C.RESET}\n")

    recognizer = None
    generator  = None
    last_captions = None
    last_prompts  = None
    output_dir = Path("visionforge_output")
    output_dir.mkdir(exist_ok=True)

    ok(f"Output directory: {output_dir.resolve()}")

    COMMANDS = {
        "analyze  <path>": "Analyze an image file",
        "generate [prompt]": "Generate from last analysis (optional extra prompt)",
        "prompt   <text>": "Set a custom generation prompt",
        "steps    <n>":    "Set inference steps (default 30)",
        "cfg      <n>":    "Set CFG scale (default 7.5)",
        "seed     <n>":    "Set random seed",
        "keywords":        "Show extracted keywords",
        "load-rec":        "Load recognition model",
        "load-gen":        "Load generation model",
        "history":         "Show training data log",
        "help":            "Show this help",
        "quit":            "Exit",
    }

    state = {"steps": 30, "cfg": 7.5, "seed": None, "custom_prompt": ""}

    def show_help():
        hdr("Commands")
        for cmd, desc in COMMANDS.items():
            print(f"  {C.CYAN}{cmd:<22}{C.RESET}{C.GRAY}{desc}{C.RESET}")

    show_help()

    while True:
        try:
            raw = input(f"\n{C.PURPLE}visionforge{C.RESET} {C.WHITE}›{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}Goodbye!{C.RESET}")
            break

        if not raw:
            continue
        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print(f"\n{C.GRAY}Goodbye!{C.RESET}")
            break

        elif cmd == "help":
            show_help()

        elif cmd == "load-rec":
            try:
                recognizer = load_recognizer()
            except Exception as e:
                err(str(e))

        elif cmd == "load-gen":
            try:
                generator = load_generator()
            except Exception as e:
                err(str(e))

        elif cmd == "analyze":
            if not arg:
                err("Usage: analyze <image_path>")
                continue
            if not Path(arg).exists():
                err(f"File not found: {arg}")
                continue
            if recognizer is None:
                info("Loading recognition model first…")
                try:
                    recognizer = load_recognizer()
                except Exception as e:
                    err(str(e)); continue

            try:
                proc, model, device = recognizer
                last_captions, size = recognize_image(arg, proc, model, device)
                hdr("Captions")
                for k, v in last_captions.items():
                    label(f"{k}:", textwrap.fill(v, 60, subsequent_indent=" "*20))
                kws = extract_keywords(last_captions)
                print(f"\n  {C.YELLOW}Keywords:{C.RESET} {C.GRAY}{', '.join(kws)}{C.RESET}")
            except Exception as e:
                err(str(e))

        elif cmd == "keywords":
            if last_captions is None:
                warn("No analysis yet — run 'analyze <path>' first")
            else:
                kws = extract_keywords(last_captions)
                print(f"\n  {C.YELLOW}Keywords:{C.RESET}")
                for kw in kws:
                    print(f"    {C.CYAN}• {kw}{C.RESET}")

        elif cmd == "prompt":
            state["custom_prompt"] = arg
            ok(f"Custom prompt set: {arg[:60]}")

        elif cmd == "steps":
            try:
                state["steps"] = int(arg)
                ok(f"Steps → {state['steps']}")
            except ValueError:
                err("Steps must be an integer")

        elif cmd == "cfg":
            try:
                state["cfg"] = float(arg)
                ok(f"CFG scale → {state['cfg']}")
            except ValueError:
                err("CFG must be a number")

        elif cmd == "seed":
            try:
                state["seed"] = int(arg) if arg else None
                ok(f"Seed → {state['seed']}")
            except ValueError:
                err("Seed must be an integer")

        elif cmd == "generate":
            extra = arg or state["custom_prompt"]
            if last_captions is None:
                warn("No analysis yet — run 'analyze <path>' first, or set a custom prompt with 'prompt <text>'")
                if not extra:
                    continue
                # Allow generation from custom prompt alone
                last_captions = {"caption": extra}

            if generator is None:
                info("Loading generation model first…")
                try:
                    generator = load_generator()
                except Exception as e:
                    err(str(e)); continue

            try:
                last_prompts = build_generation_prompt(last_captions, extra)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = str(output_dir / f"generated_{ts}.png")

                pipe, device = generator
                generate_image(
                    pipe, last_prompts, out_path,
                    steps=state["steps"],
                    cfg=state["cfg"],
                    seed=state["seed"]
                )

                save_training_entry(
                    last_captions, last_prompts,
                    extra or "manual",
                    output_dir
                )

                hdr("Result")
                label("Output:", out_path)
                label("Prompt:", last_prompts["positive"][:70] + "…")
                print(f"\n  {C.GREEN}Open {out_path} to view your image!{C.RESET}")

            except Exception as e:
                err(str(e))

        elif cmd == "history":
            log = output_dir / "training_data.jsonl"
            if not log.exists():
                warn("No training data yet")
            else:
                hdr("Training Data Log")
                with open(log) as f:
                    entries = [json.loads(l) for l in f if l.strip()]
                print(f"  {C.GRAY}{len(entries)} entries in {log}{C.RESET}\n")
                for e in entries[-5:]:
                    label(e["id"], e["timestamp"])

        else:
            warn(f"Unknown command: '{cmd}' — type 'help' for commands")


# ── CLI / batch mode ──────────────────────────────────────────────────────────

def batch_mode(args):
    """Non-interactive: analyze + generate in one shot."""
    banner()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze
    proc, model, device = load_recognizer()
    captions, size = recognize_image(args.image, proc, model, device)

    hdr("Captions")
    for k, v in captions.items():
        label(f"{k}:", v)

    prompts = build_generation_prompt(captions, args.prompt or "")

    hdr("Built Prompt")
    label("Positive:", prompts["positive"][:80] + "…")
    label("Negative:", prompts["negative"][:60] + "…")

    # Generate
    pipe, device = load_generator()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(output_dir / f"generated_{ts}.png")
    generate_image(pipe, prompts, out_path, steps=args.steps, cfg=args.cfg)

    save_training_entry(captions, prompts, args.image, output_dir)

    hdr("Done")
    label("Generated:", out_path)
    label("Training log:", str(output_dir / "training_data.jsonl"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_deps()

    parser = argparse.ArgumentParser(
        description="VisionForge — Local image recognition + SD 1.5 generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Examples:
            # Interactive mode (recommended)
            python visionforge.py

            # Batch: analyze photo.jpg and generate a variation
            python visionforge.py --image photo.jpg

            # Batch with custom prompt
            python visionforge.py --image photo.jpg --prompt "cinematic sunset"

            # More steps for higher quality
            python visionforge.py --image photo.jpg --steps 50
        """)
    )
    parser.add_argument("--image",      help="Input image path (batch mode)")
    parser.add_argument("--prompt",     help="Extra generation prompt text", default="")
    parser.add_argument("--steps",      type=int,   default=30, help="Inference steps (default: 30)")
    parser.add_argument("--cfg",        type=float, default=7.5, help="CFG scale (default: 7.5)")
    parser.add_argument("--output-dir", default="visionforge_output", help="Output directory")

    args = parser.parse_args()

    if args.image:
        batch_mode(args)
    else:
        interactive_mode(None, None)
