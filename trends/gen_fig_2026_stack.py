#!/usr/bin/env python3
"""Generate the 2026 LLM-systems stack diagram using Gemini image generation."""
import os, sys, time
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable.")
    print("  Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

MODEL = "gemini-3-pro-image-preview"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
client = genai.Client(api_key=API_KEY)

PROMPT = """
Create a MODERN MINIMAL-style layered architecture diagram for a computer-systems
research talk, titled "The 2026 LLM Systems Stack".
It shows six stacked horizontal layers, four vertical cross-cutting concern rails on the
right, and one feedback loop on the left. It must feel authoritative, calm and precise.
Landscape orientation, 16:9, designed to be projected on a large screen.

VISUAL STYLE - MODERN MINIMAL:
- Ultra-clean geometric shapes with crisp edges.
- Each of the six layers is a full-width rounded rectangle (14px radius) filled with a
  flat desaturated color block. NOT a thin accent bar, but a full fill.
- Inside each layer, the technique items are white rounded pill boxes (999px radius,
  i.e. fully rounded ends), NO visible border, floating on the layer fill with a subtle
  shadow (1px offset, 4px blur, rgba(0,0,0,0.06)), arranged in one horizontal row,
  evenly spaced.
- Each layer has a left-hand label block: a bold layer name in small caps plus a
  smaller gray one-line description underneath.
- Typography: Inter or a similar geometric sans-serif. Layer names 700 weight and
  letter-spaced. Pill text 400 weight, small. Title 700 weight.
- Labels are INSIDE their boxes, never floating beside them.
- Generous whitespace: at least 20px between layers and between pills.
- Pure white page background (#FFFFFF).
- NO icons, NO clip art, NO photographs, NO 3D perspective, NO gradients other than the
  flat layer fills. Let the structure speak.

COLOR PALETTE (use these exact hex values):
- Page background: #FFFFFF
- Layer 6 fill #F3EDFB accent #5E60CE
- Layer 5 fill #E4F0EC accent #2A9D8F
- Layer 4 fill #FDF3E3 accent #D9A02A
- Layer 3 fill #E8EDF2 accent #264653
- Layer 2 fill #FBEDE4 accent #E76F51
- Layer 1 fill #F1F1EF accent #6B7280
- Pill fill #FFFFFF, body text #2C2C2C, secondary text #6B7280
- Cross-cutting rails fill #FAFAF8 with a 3px left edge in their own accent color

LAYOUT:
- Bold left-aligned title at the top: "The 2026 LLM Systems Stack"
  with a smaller gray subtitle directly beneath:
  "what 2026 papers build, and the concerns that cut through every layer"
- Below the title, six full-width layers stacked vertically. Layer 6 is at the TOP and
  layer 1 is at the BOTTOM. Each layer occupies one horizontal band of equal height.
- The right 18 percent of the layer stack is reserved for four narrow VERTICAL rails
  that span the full height of all six layers, side by side. Each rail contains one
  label written horizontally in small stacked lines, NOT rotated.
- The left 8 percent outside the layers is reserved for one feedback loop arrow.

LAYER CONTENTS - spell every word EXACTLY as written.
Layer name, then its gray one-line description, then the pills in left-to-right order.

Layer 6 name: "AGENT RUNTIME"
Layer 6 description: "the new unit of work is a trajectory, not a request"
Layer 6 pills: "Agent memory and KV time-to-live" | "Declarative agent specs" |
  "Async tool-call concurrency" | "Computer-use agents" | "Structured output engines"

Layer 5 name: "RL POST-TRAINING PLANE"
Layer 5 description: "training became an online serving workload"
Layer 5 pills: "Disaggregated rollout and training" | "Async off-policy pipelines" |
  "Speculative drafters in rollout" | "Preemptible and spot harvesting" |
  "Role-based fault tolerance"

Layer 4 name: "SERVING RUNTIME"
Layer 4 description: "goodput under multi-SLO, multi-tenant load"
Layer 4 pills: "Prefill-decode and EPD disaggregation" | "Hierarchical KV cache pools" |
  "Speculative decoding" | "SLO and goodput scheduling" | "Elastic autoscaling and cold start"

Layer 3 name: "PARALLELISM AND FLEET"
Layer 3 description: "the cluster is no longer homogeneous"
Layer 3 pills: "Heterogeneous auto-parallelism" | "MoE expert parallelism" |
  "Context and sequence parallelism" | "Cross-cluster and cross-region" |
  "Spot and decentralized capacity"

Layer 2 name: "COMPILER AND KERNEL"
Layer 2 description: "the fastest-growing layer of 2026"
Layer 2 pills: "Megakernels" | "Tile-level languages" | "Sparse attention kernels" |
  "Compute-communication overlap" | "CUDA graph and runtime codegen"

Layer 1 name: "HARDWARE SUBSTRATE"
Layer 1 description: "asymmetric, memory-starved, power-limited"
Layer 1 pills: "Blackwell-class GPUs and superchips" | "PIM, CXL and near-data memory" |
  "SRAM-resident pipelines" | "RDMA and multipath fabric" | "Mobile NPUs"

CROSS-CUTTING RAILS - four narrow vertical bands on the right, spanning all six layers,
each with a small caps title on its own lines, plus one short gray line underneath:
Rail 1 title: "ENERGY AND POWER"        Rail 1 note: "joules per token"
Rail 2 title: "RELIABILITY AND SILENT DATA CORRUPTION"   Rail 2 note: "correctness, not just uptime"
Rail 3 title: "COST AND ELASTICITY"     Rail 3 note: "dollars per goodput"
Rail 4 title: "MEASUREMENT AND SKEPTICISM"   Rail 4 note: "does the trick actually help"
Rail 1 accent #E9C46A, Rail 2 accent #E76F51, Rail 3 accent #2A9D8F, Rail 4 accent #264653.

FEEDBACK LOOP on the left, outside the layer stack:
- One thin (2px) gray (#6B7280) curved arrow that starts at the left edge of Layer 6,
  sweeps out to the left, and ends with a clean arrowhead at the left edge of Layer 2.
- A small white rounded label box sits on that curve, reading exactly:
  "AI-FOR-SYSTEMS FLYWHEEL"
- A second, thinner curved arrow returns from the left edge of Layer 2 back up to the
  left edge of Layer 6, drawn just inside the first curve.
- Two tiny gray captions along the curves, each in its own small white rounded box:
  on the downward curve: "agents generate kernels and configs"
  on the upward curve: "benchmarks verify and redeploy"

CONNECTIONS:
- Draw one thin vertical arrow (1.5px, #6B7280) in the middle gap between each pair of
  adjacent layers, pointing DOWNWARD, from layer 6 to 5, 5 to 4, 4 to 3, 3 to 2, 2 to 1.
- Five such inter-layer arrows in total, all identical, all vertical, all centered.
- No arrows between the pills. No arrows crossing the rails.

CONSTRAINTS:
- SPELL EVERY LABEL EXACTLY as given. Do not invent, translate, abbreviate, reorder,
  or drop any listed pill. Technical terms such as "PIM, CXL and near-data memory",
  "Prefill-decode and EPD disaggregation", "KV time-to-live", "Megakernels",
  "CUDA graph and runtime codegen", "Blackwell-class GPUs and superchips" must appear
  verbatim.
- Exactly six layers and exactly five pills per layer. Do not add a seventh layer.
- No emoji, no icons, no logos, no clip art, no photorealistic elements.
- No gradients, no glow, no neon, no dark mode, no 3D isometric view.
- No fake chart axes, no invented numbers, no watermark, no signature.
- All text must be crisp and horizontal. NEVER rotate any text, including the rails.
- Keep the page background pure white and do not add a border frame around the image.
"""


def generate_image(prompt_text, attempt_num):
    print(f"\n{'='*60}\nAttempt {attempt_num}\n{'='*60}")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt_text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        output_path = os.path.join(OUTPUT_DIR, f"fig_2026_stack_attempt{attempt_num}.png")
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
                return output_path
            elif part.text:
                print(f"Text: {part.text[:300]}")
        print("WARNING: No image in response")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    results = []
    for i in range(1, 4):
        if i > 1:
            time.sleep(2)
        path = generate_image(PROMPT, i)
        if path:
            results.append(path)
    if not results:
        print("All attempts failed!")
        sys.exit(1)
    print(f"\nGenerated {len(results)} attempts. Review and pick the best.")


if __name__ == "__main__":
    main()


