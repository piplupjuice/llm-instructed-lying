"""Write gallery_runs.tex: every per-run diagnostic figure from the pipeline, grouped by figure type."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [("llama_syn", "Llama-8B", "Synthetic"), ("llama_gsm", "Llama-8B", "GSM8K"), ("llama_math", "Llama-8B", "MATH500"),
     ("qwen_syn", "Qwen-3B", "Synthetic"), ("qwen_gsm", "Qwen-3B", "GSM8K"), ("qwen_math", "Qwen-3B", "MATH500"),
     ("gemma_syn", "Gemma-2B", "Synthetic"), ("gemma_gsm", "Gemma-2B", "GSM8K"), ("gemma_math", "Gemma-2B", "MATH500"),
     ("qmath_syn", "Qwen-Math", "Synthetic"), ("qmath_gsm", "Qwen-Math", "GSM8K"), ("qmath_math", "Qwen-Math", "MATH500")]
out = []


def grid(fname, title, cap, label, ncol, per_page):
    runs = [r for r in R if os.path.exists(f"{ROOT}/figures/runs/{r[0]}/{fname}.pdf")]
    w = {1: "1.0", 2: "0.495", 3: "0.325"}[ncol]
    for ci in range(0, len(runs), per_page):
        chunk = runs[ci:ci + per_page]
        out.append(r"\begin{figure}[p]\centering")
        for j, (r, m, d) in enumerate(chunk):
            out.append(r"\begin{minipage}[t]{" + w + r"\textwidth}\centering{\scriptsize\textbf{" + m + r"}, \textsc{" + d + r"}}\\[1pt]"
                       r"\includegraphics[width=\linewidth]{figures/runs/" + r + "/" + fname + r".pdf}\end{minipage}"
                       + (r"\hfill" if (j + 1) % ncol else r"\par\vspace{5pt}"))
        first = ci == 0
        text = cap if first else "Continued; see the first part of this figure for how to read the panels."
        out.append(r"\caption{\textbf{" + title + ("" if first else " (continued)") + r".} " + text + "}" + (r"\label{" + label + "}" if first else ""))
        out.append(r"\end{figure}")
        out.append("")


grid("fig_lie_shift_P2", r"Lie-shift profiles at P2, all 12 runs",
     r"Left: relative shift size $\lVert\hlie-\htruth\rVert/\lVert\htruth\rVert$ by depth. Middle: cosine of each item's shift with the held-out mean lie direction. Right: projection on a held-out truth direction under the lie prompt. Lines: outcome means with 95\% intervals; ticks: layers where \lie{} and \leak{} differ (FDR $q<.05$). In every run with enough \leak{} items, the \leak{} size curve drops below \lie{} in the last part of the network while its alignment stays as high or higher: the same direction with a smaller length, the observation that motivated the \pll/\prp{} decomposition of \S\ref{sec:geometry}.",
     "fig:gal-shiftP2", 2, 8)
grid("fig_lie_shift_P1", r"Lie-shift profiles at P1, all 12 runs",
     r"Same panels as Figure~\ref{fig:gal-shiftP2}, at the last prompt token. All four outcomes rise together to a common plateau and align with one shared direction: before generation, the instruction moves every item alike.",
     "fig:gal-shiftP1", 2, 8)
grid("fig_softmax", r"Logit lens and answer-token statistics, all 12 runs",
     r"Left: logit-lens rank of the gold answer's first token by depth, as $\log_{10}(\text{rank}+1)$ (solid: lie prompt; dotted: truth prompt; 0 = top). Middle: log-probability of the full gold answer under the lie prompt (near 0 for \leak{} and \para{} by construction). Right: next-token entropy at P2. \lie{} items reach the gold answer under the truth prompt and suppress it under the lie prompt; \leak{} and \para{} reach it under the lie prompt; \ign{} items reach it under neither. Entropy is near zero everywhere: the answer is chosen with near certainty.",
     "fig:gal-softmax", 2, 8)
grid("fig_probe", r"Layer-wise \lie{} vs.\ \leak{} probes, all runs where both classes have at least 15 items",
     r"Black: held-out AUROC from lie-prompt states; purple dotted: truth-prompt control (same labels, states without any lie instruction); grey band: 95\% label-permutation null; ticks: layers significant after correction. At P1 the probe never clearly beats its control; at P2 it does in every run. Qwen-3B and Qwen-Math \textsc{Synthetic} are inflated by outputs that box two answers.",
     "fig:gal-probe", 2, 10)
grid("fig_umap_P2", r"UMAP of hidden states at P2, all 12 runs",
     r"Rows: both prompts (circles truth, triangles lie), lie prompt only, and lie shift; columns: 3, 25, 50, 75, 100\% of depth. Titles give the silhouette by outcome with a permutation $p$. Truth and lie prompts form separate islands, but outcomes overlap almost everywhere (silhouette near 0), even where probes reach AUROC 0.8--0.9: the lie/leak difference lives in a narrow direction, not in the bulk geometry. The clearest exception is the final layer of the Qwen models, where the upcoming answer dominates.",
     "fig:gal-umapP2", 2, 6)
grid("fig_umap_P1", r"UMAP of hidden states at P1, all 12 runs",
     r"As Figure~\ref{fig:gal-umapP2}, at the last prompt token. Outcomes are fully mixed: nothing about the eventual outcome is visible before generation.",
     "fig:gal-umapP1", 2, 6)
grid("fig_geometry_4_questions_P2", r"Four-question geometry at P2, all 12 runs",
     r"One random item per outcome. (a) Cosines among the 8 states (4 items $\times$ 2 prompts) at a late layer. (b) PCA map, arrows truth$\to$lie; grey: all other items. (c) Per item, cosine between truth and lie states (solid) and shift size (dashed; the thin line at $\sqrt2$ is the size expected for unrelated vectors). (d) Cosine between the shifts of two items. Single items are illustrations, not evidence; the population statistics are in \S\ref{sec:geometry}.",
     "fig:gal-4q", 2, 4)
grid("fig_geometry_4_questions_P1", r"Four-question geometry at P1 (runs where it was produced)",
     r"As Figure~\ref{fig:gal-4q}, at the last prompt token.",
     "fig:gal-4qP1", 2, 4)
grid("fig_geometry_4_questions_3d_P2", r"Layer-by-layer paths of the four example items at P2, all 12 runs",
     r"Each point is one layer in a 3D PCA; solid: truth prompt, dashed: lie prompt. Truth paths stay in a tight bundle; lie paths travel much further.",
     "fig:gal-4q3d", 3, 9)
grid("fig_geometry_4_questions_3d_P1", r"Layer-by-layer paths of the four example items at P1",
     r"As Figure~\ref{fig:gal-4q3d}, at the last prompt token.",
     "fig:gal-4q3dP1", 3, 9)
grid("fig_attention_P2_question", r"Head-level attention from P2 to the question, all 12 runs",
     r"First four maps: mean attention per (layer, head) for \lie, \leak, \ign, \para{} (up to 60 items each); last map: \lie$-$\leak, with $\times$ marking cells significant after FDR correction. The difference is run-specific (338 cells in Llama \textsc{GSM8K}, none in Llama \textsc{MATH500}), so we treat attention as exploratory.",
     "fig:gal-attq", 1, 6)
grid("fig_attention_P2_reasoning", r"Head-level attention from P2 to the model's own reasoning, all 12 runs",
     r"As Figure~\ref{fig:gal-attq}, for attention to the generated reasoning.",
     "fig:gal-attr", 1, 6)
grid("fig_attention_P2_instruction", r"Head-level attention from P2 to the instruction, all 12 runs",
     r"As Figure~\ref{fig:gal-attq}, for attention to the instruction text.",
     "fig:gal-atti2", 1, 6)
grid("fig_attention_P1_instruction", r"Head-level attention from P1 to the instruction, all 12 runs",
     r"As Figure~\ref{fig:gal-attq}, from the last prompt token to the instruction text.",
     "fig:gal-atti1", 1, 6)
grid("fig_attention_P1_question", r"Head-level attention from P1 to the question, all 12 runs",
     r"As Figure~\ref{fig:gal-attq}, from the last prompt token to the question.",
     "fig:gal-attq1", 1, 6)
for c in ("C1", "C2", "C3", "C4"):
    grid(f"fig_tokens_P1_{c}", r"Token-level attention at P1 for one \textbf{" + c + r"} example per run",
         r"Mean over layers and heads, last 80 prompt tokens. Attention sits mostly on the chat-template tokens that end the prompt.",
         f"fig:gal-tokP1{c}", 2, 12)
for c in ("C1", "C2", "C3", "C4"):
    grid(f"fig_tokens_P2_{c}", r"Token-level attention at P2 for one \textbf{" + c + r"} example per run",
         r"Mean over layers and heads, last 80 tokens before the answer. Attention concentrates on \texttt{\textbackslash boxed\{} and the tokens just before it, typically the result the model has just computed in its own reasoning.",
         f"fig:gal-tok{c}", 1, 12)
open(f"{ROOT}/gallery_runs.tex", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("figures:", sum(1 for x in out if x.startswith(r"\begin{figure}")))
