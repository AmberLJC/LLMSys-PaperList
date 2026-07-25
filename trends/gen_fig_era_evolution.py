#!/usr/bin/env python3
"""Generate the three-era evolution diagram (2024 -> 2026) using Gemini image generation."""
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
Create a MODERN MINIMAL-style technical diagram for a computer-systems research talk.
The diagram is a 3-column comparison matrix titled "Three Eras of LLM Systems Research".
It must feel authoritative, calm, precise and information-dense but uncluttered.
Landscape orientation, 16:9, designed to be projected on a large screen.

VISUAL STYLE - MODERN MINIMAL:
- Ultra-clean geometric shapes with crisp edges.
- Each of the three era columns is a large full-height rounded rectangle (16px radius)
  filled with a bold desaturated color block. NOT a thin accent bar, but a full fill.
- Component boxes inside the columns have ROUNDED CORNERS (12px radius), white fill,
  NO visible border, floating on the column fill with a subtle shadow
  (1px offset, 4px blur, rgba(0,0,0,0.06)).
- ONE accent color per column, used only on the column header text and the header rule.
- Thin (1.5px) dark gray (#6B7280) arrows, small clean arrowheads. No thick colored arrows.
- Typography: Inter or a similar geometric sans-serif. Column headers 700 weight and
  letter-spaced small caps. Row labels 600 weight. Body text 400 weight.
- Labels are INSIDE boxes, never floating beside them.
- Generous whitespace, at least 24px between elements.
- Pure white page background (#FFFFFF).
- NO icons, NO decorative elements, NO clip art, NO photographs, NO 3D effects,
  NO gradients other than the flat column fills. Let the structure speak.

COLOR PALETTE (use these exact hex values):
- Page background: #FFFFFF
- Column 1 fill: #E8EDF2   Column 1 accent: #264653
- Column 2 fill: #E4F0EC   Column 2 accent: #2A9D8F
- Column 3 fill: #FBEDE4   Column 3 accent: #E76F51
- Box fill: #FFFFFF        Body text: #2C2C2C        Secondary text: #6B7280
- Bottom band fill: #F5F5F3

LAYOUT:
- A single bold left-aligned title across the top: "Three Eras of LLM Systems Research"
  and directly under it a smaller gray subtitle:
  "what each year's papers were actually optimizing"
- Below the title, a 4-row by 3-column matrix.
- The far-left margin holds four right-aligned ROW LABELS in dark gray small caps,
  vertically centered against their row:
  Row 1 label: "UNIT OF WORK"
  Row 2 label: "MAIN BOTTLENECK"
  Row 3 label: "HERO TECHNIQUES"
  Row 4 label: "REPRESENTATIVE SYSTEMS"
- The three columns are equal width, side by side, separated by whitespace.
- Each column has a header at its top: a large year, then an em dash, then an era name:
  Column 1 header: "2024 - SCALE OUT"
  Column 2 header: "2025 - DISAGGREGATE AND REASON"
  Column 3 header: "2026 - AGENT-NATIVE AND ACCOUNTABLE"
  Under each header draw a 3px horizontal rule in that column's accent color.
- Each cell of the matrix is one white rounded box containing short text.
  Rows 1 and 2 hold one short sentence. Rows 3 and 4 hold a compact bulleted list,
  one item per line, small text, with a tiny accent-colored dot as the bullet.

CELL CONTENTS - spell every word EXACTLY as written:

Column 1, Row 1: "One request"
Column 1, Row 2: "GPU memory capacity and batching efficiency"
Column 1, Row 3 (bulleted list):
  "3D parallelism at 10,000 GPUs"
  "Continuous batching"
  "PagedAttention"
  "Chunked prefill"
  "Prefill-decode disaggregation v1"
  "Checkpoint and restart recovery"
Column 1, Row 4 (bulleted list):
  "MegaScale"
  "DistServe"
  "Sarathi-Serve"
  "Llumnix"
  "ByteCheckpoint"

Column 2, Row 1: "A session or a reasoning trace"
Column 2, Row 2: "Long context, KV cache capacity, rollout long tail"
Column 2, Row 3 (bulleted list):
  "MoE expert parallelism"
  "Compute-communication overlap"
  "Context parallelism to 2M tokens"
  "Shared KV cache pools"
  "Asynchronous RL rollout"
  "Quantized attention"
Column 2, Row 4 (bulleted list):
  "ByteScale"
  "Comet"
  "MegaScale-Infer"
  "AReaL"
  "TokenLake"
  "Autellix"

Column 3, Row 1: "A whole agent trajectory"
Column 3, Row 2: "Cross-request state, fleet heterogeneity, joules, silent faults"
Column 3, Row 3 (bulleted list):
  "Agent memory and KV time-to-live"
  "Native sparse attention"
  "FP8 and FP4 numerics"
  "Megakernel compilers"
  "Speculative decoding everywhere"
  "Cross-cluster and spot fleets"
  "Silent data corruption detection"
  "Energy-aware DVFS"
Column 3, Row 4 (bulleted list):
  "Tessera"
  "Weave and RLinf"
  "KAIROS"
  "Hippocampus"
  "MPK"
  "OpGuard"
  "VoltanaLLM"

- At the bottom, spanning the full width, a light gray rounded band (#F5F5F3)
  with the small caps heading "FOUR SHIFTS THAT CUT ACROSS ALL THREE YEARS"
  and four equal white rounded chips in a single row, each holding one line of text:
  Chip 1: "Request -> Session -> Agent trajectory"
  Chip 2: "Homogeneous cluster -> Heterogeneous, spot, cross-region fleet"
  Chip 3: "Throughput -> Goodput per SLO -> Goodput per joule per dollar"
  Chip 4: "Humans write systems -> Agents write kernels, benchmarks close the loop"

CONNECTIONS:
- Draw one thin horizontal arrow (1.5px, #6B7280, small clean arrowhead) between the
  header of Column 1 and the header of Column 2, pointing right.
- Draw an identical arrow between the header of Column 2 and the header of Column 3.
- These two arrows are the only arrows in the upper matrix. They convey time order.
- Inside the four bottom chips, render the "->" as a small thin gray arrow glyph.
- No other connectors, no lines between rows, no boxes around the row labels.

CONSTRAINTS:
- SPELL EVERY LABEL EXACTLY as given. Do not invent, translate, abbreviate, reorder,
  or drop any listed item. Product names such as "PagedAttention", "Sarathi-Serve",
  "MegaScale-Infer", "ByteCheckpoint", "VoltanaLLM", "RLinf", "AReaL", "TokenLake",
  "Autellix", "OpGuard", "KAIROS", "Hippocampus", "Tessera", "MPK", "Comet",
  "ByteScale", "DistServe", "Llumnix", "MegaScale", "Weave" must appear verbatim.
- No emoji, no icons, no logos, no clip art, no photorealistic elements.
- No gradients, no glow, no drop shadows other than the specified subtle box shadow.
- No fake chart axes, no random numbers, no watermark, no signature.
- All text must be crisp, horizontal, and left-aligned inside its box. Never rotate text.
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
        output_path = os.path.join(OUTPUT_DIR, f"fig_era_evolution_attempt{attempt_num}.png")
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
