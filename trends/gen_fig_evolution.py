#!/usr/bin/env python3
"""Generate a Modern-Minimal conceptual diagram showing how LLM-systems research
has evolved 2024 -> 2025 -> 2026, using Gemini 3 Pro Image Preview.

Run: GEMINI_API_KEY=... python3 figures/gen_fig_evolution.py
Outputs 3 attempts: figures/fig_evolution_attempt{1,2,3}.png
"""
import os
import sys
import time

try:
    from google import genai
except ImportError:
    print("Install: pip install google-genai")
    sys.exit(1)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable.")
    print("  Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

MODEL = "gemini-3-pro-image-preview"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
client = genai.Client(api_key=API_KEY)

PROMPT = """
1. FRAMING
Create a Modern-Minimal-style technical diagram for an academic systems paper
(MLSys/EuroSys-tier). The diagram should feel confident, clean, and authoritative.
It is a single landscape figure summarizing how Large Language Model systems
research evolved across three distinct eras: 2024, 2025, and 2026. Width:height
= 16:9. Pure-white background.

2. VISUAL STYLE - MODERN MINIMAL
- Ultra-clean geometric shapes with crisp edges.
- Three full-height vertical SECTION COLUMNS as backgrounds, each filled with a
  desaturated tone covering its column completely:
    Column 1 (2024): slate blue background #E8EDF2
    Column 2 (2025): warm sand background #F5F0E8
    Column 3 (2026): cool mint background #E8F2EE
- Inside each column, every concept box has ROUNDED CORNERS (12px radius),
  white fill (#FFFFFF), NO visible border, but a subtle drop shadow
  (1px offset, 4px blur, rgba(0,0,0,0.06)).
- ONE accent color per column, used sparingly on the column header bar and on
  one "hero" element:
    2024 accent: Deep blue   #2563EB
    2025 accent: Amber       #D97706
    2026 accent: Emerald     #059669
- Connections between columns are thin (1.5px) dark-gray arrows (#6B7280)
  with a small filled circle at source and a clean arrowhead at target.
- Typography: clean sans-serif (Inter look). Section headers 600 weight in
  the column's accent color. Body text 400 weight in dark gray #374151.
- Labels live INSIDE boxes, not beside them. Generous whitespace, at least
  24px between elements.
- NO decorative elements, NO clip-art icons, NO emoji.

3. COLOR PALETTE (exact hex)
Backgrounds: #E8EDF2 (slate), #F5F0E8 (sand), #E8F2EE (mint)
Accents:     #2563EB (blue), #D97706 (amber), #059669 (emerald)
Boxes:       #FFFFFF fill, shadow rgba(0,0,0,0.06)
Text:        #111827 (headers), #374151 (body), #6B7280 (annotations)
Arrows:      #6B7280 (1.5px)
Hero number: column accent color, 600 weight, 28pt

4. LAYOUT (exact, exhaustive)
Three columns of equal width spanning the canvas. Title bar across the top:

TOP TITLE BAR (centered, 18pt, color #111827, weight 700):
"Three Eras of LLM Systems Research"

SUBTITLE (centered, 11pt, color #6B7280, italic, just below title):
"From training-centric to serving-dominant to agent and multimodal"

Below the title, each column has:

COLUMN HEADER (full-width inside the column):
  - Large era label, 24pt, weight 600, in column accent color
  - Below it, a one-line tagline in #374151, 11pt

Then 4 STACKED WHITE BOXES per column, each:
  - 80px tall, full column width minus 24px padding
  - Top-left corner has a small tag in accent color, 9pt, all caps:
    e.g., "DOMINANT TOPIC"
  - Below the tag, a one-line claim, 12pt, #111827, weight 500
  - Below the claim, a short data callout, 10pt, #6B7280

Exact content per column (SPELL EXACTLY, do not paraphrase):

== COLUMN 1: 2024 ==
HEADER: "2024"
TAGLINE: "Training-centric era"
BOX 1 — Tag: "DOMINANT TOPIC" | Claim: "Pre-training at scale" | Callout: "17 papers - MegaScale, Llama 3, DeepSeek-V3"
BOX 2 — Tag: "INFRASTRUCTURE" | Claim: "10000+ GPU clusters" | Callout: "Fault tolerance peaks: 8 papers"
BOX 3 — Tag: "INDUSTRY" | Claim: "Peak technical-report year" | Callout: "46 reports released"
BOX 4 — Tag: "SERVING" | Claim: "Disaggregation emerges" | Callout: "DistServe, Sarathi, Llumnix"

== COLUMN 2: 2025 ==
HEADER: "2025"
TAGLINE: "Serving-dominant era"
BOX 1 — Tag: "DOMINANT TOPIC" | Claim: "LLM serving overtakes training" | Callout: "54 serving papers, only 22 pre-training"
BOX 2 — Tag: "POST-TRAINING" | Claim: "RLHF infrastructure peaks" | Callout: "10 papers - AReaL, StreamRL, RhymeRL"
BOX 3 — Tag: "MULTIMODAL" | Claim: "Multi-modal serving 3.5x" | Callout: "From 4 papers to 14 papers"
BOX 4 — Tag: "REASONING" | Claim: "DeepSeek-R1 era begins" | Callout: "Test-time scaling, long-thought serving"

== COLUMN 3: 2026 ==
HEADER: "2026"
TAGLINE: "Agent and multimodal era"
BOX 1 — Tag: "DOMINANT TOPIC" | Claim: "Agentic systems and disaggregation" | Callout: "12 agent papers - 3x growth"
BOX 2 — Tag: "BREAKOUT" | Claim: "Edge LLM inference takes off" | Callout: "5x jump - 10 papers in 2026"
BOX 3 — Tag: "VENUE" | Claim: "MLSys'26 + EuroSys'26 surge" | Callout: "132 systems-LLM papers across both"
BOX 4 — Tag: "ALGORITHM-SYSTEM CO-DESIGN" | Claim: "Speculative decoding everywhere" | Callout: "9x jump in 2 years"

5. CONNECTIONS (between columns)
Thin gray arrows (#6B7280, 1.5px) connecting evolution across eras:

ARROW A: from "Pre-training at scale" (Col 1, Box 1) -> "LLM serving overtakes training" (Col 2, Box 1)
  Label above arrow midpoint, 9pt italic #6B7280: "models stabilize"

ARROW B: from "RLHF infrastructure peaks" (Col 2, Box 2) -> "Agentic systems and disaggregation" (Col 3, Box 1)
  Label: "reasoning drives agents"

ARROW C: from "Disaggregation emerges" (Col 1, Box 4) -> "Speculative decoding everywhere" (Col 3, Box 4)
  Long curved arrow arcing across the figure, label: "decode latency wars"

BOTTOM BAR (10px tall, full width across all columns, just under the lowest box row):
Solid bar in light gray #F3F4F6 with text centered:
"Data from LLMSys-PaperList - 3-year trend, 2024 to 2026" - 9pt #6B7280

6. CONSTRAINTS
- NO stock icons, NO clip art, NO emoji, NO decorative graphics.
- NO 3D effects, NO gradients beyond a single column-background tone.
- DO NOT change any text. SPELL EVERY LABEL EXACTLY as listed above.
- DO NOT add boxes I did not list. EXACTLY 4 boxes per column, 3 columns.
- Maintain consistent box heights and widths within each column.
- White background outside the three columns - the colored fills are only
  inside the column rectangles.
- The figure must read left-to-right as a clear evolution: 2024 -> 2025 -> 2026.
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
        output_path = os.path.join(OUTPUT_DIR, f"fig_evolution_attempt{attempt_num}.png")
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
    print(f"\nGenerated {len(results)} attempts. Review them under figures/ and "
          f"copy the best to fig_evolution.png")


if __name__ == "__main__":
    main()
