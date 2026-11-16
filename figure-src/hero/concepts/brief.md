# Common brief: README hero concept for `auditable`

You are generating a whole-figure VISUAL CONCEPT image (not code, not a diagram drawn with matplotlib/SVG). Use your image-generation capability (for example a `generate_image` tool, Gemini image model, "nano banana", or whatever image tool your runtime exposes). Save the generated image as PNG at the output path given in your unit prompt. If your runtime has NO image-generation capability, do not draw it with code: write that in the result file and stop.

## What the figure is for

- Placement: the hero image right under the header of the GitHub README (and PyPI page) of `auditable`, an open-source Python library of auditing methods for LLM agents. Displayed about 880 to 1000 px wide on a white page. Aspect ratio about 2.2:1 (wide).
- Reader takeaway (one sentence): auditable audits an LLM agent at three points, before it runs (its configuration), while it runs (each new step), and after it ends (did the run fail, and which step was the decisive mistake), with one family of methods and a leaderboard at each point.
- Reader decision: a visitor should understand in five seconds what the library audits and when, and want to try it.

## Facts (use exactly these; invent nothing else)

- PRE, before the run. Input: an agent configuration = the task text plus the declared tools and their permission levels. Example (real, from the README): task "Read the quarterly report and summarize the revenue section", tools `read_report` (read) and `run_shell` (execute). Method family `config` (8 methods: OWASP and CWE rule scanners, an LLM judge). Call `audit(config)`. Output: a finding that flags `run_shell` as an excess capability. Leaderboard `pre/over_privilege`.
- LIVE, while the run streams. Input: the steps arriving one by one (a trace: each step is an agent decision or a tool call, with dependencies on earlier steps). Family `online` (5 methods, e.g. a causal z-score of dependency span). Call `step_alarm_scores(trace)`: one alarm score per arriving step; the alarm at step t reads only steps 0..t. Leaderboards `live/swegym`, `live/tau`.
- POST, after the run. Two questions on the finished trace: (1) Did the run fail? family `run` (17 methods: PyOD and PyGOD detectors, GUARDIAN, G-Safeguard, GRADE regressions), call `decision_function(traces)` gives one failure score per run, leaderboards `detect/swegym`, `detect/tau`. (2) Which step was the decisive mistake? family `step` (14 methods: blast radius, Who&When LLM judges), call `rank(trace)` orders the steps most suspected first, leaderboard `localize/whoandwhen`.
- Totals: 44 methods, 4 families, one interface per family as in PyOD, 6 leaderboards that rerun the CatchBench boards.
- Optional small real plot (only these values): the `detect/swegym` leaderboard, ROC-AUC: GAAN 0.850, spline size control 0.832, G-Safeguard-Lite 0.828, GRADE full 0.819, GRADE flatdep 0.804, random 0.483.

## Preference mapping (the maintainer's confirmed taste; this is the top design priority)

Reference images are attached as files; open and look at them:
- `ref-14-memoharness.png` (MemoHarness Figure 1, strongly liked): imitate its polish: colored phase banners with a round icon badge, numbered sub-steps, gently shaded 3D objects (cubes, documents, small robot/LLM glyphs), rich but soft colors, compact math anchors such as `s_t = f(x_{0..t})`, dashed grouping boxes, short labels. Low "AI look", well polished.
- `ref-17-meta.png` (Meta overview, liked): attractive shallow-3D technical objects that make the mechanism self-explanatory; consistent color per role.
- `ref-12-tdc.png` (TDC overview, liked): a large readable overview that is not monotone and includes one real small plot.
Avoid: three pastel boxes full of sentences (the current version was rejected for exactly this: "not good enough"), walls of text, cartoon mascots, heavy saturated colors, random decoration, fake numbers.

## Palette and encodings

White background. Stage colors, used consistently on banners, objects and arrows: PRE gold `#ECB66C` (stroke `#B8862F`), LIVE teal `#4198AC` (stroke `#2B6E7E`), POST coral `#ED8D5A` (stroke `#B66037`). Shared structure (the trace, rails, neutral frames) in mint `#BFDFD2` with stroke `#365E49`. Text ink `#202923`, secondary `#67756D`. Code in a monospace font.

## Must show

1. The three stages PRE, LIVE, POST as the main reading order, left to right, with concrete objects that take part in the audit (a configuration card with tool chips and a flagged `run_shell`; a stream of step nodes with dependency arcs and a rising alarm curve; a finished trace with a failure score and one step highlighted as the ranked mistake).
2. For each stage, short labels only: the question, the family and its method count, the call in monospace, and the leaderboard id(s).
3. A compact footer or badge: "44 methods · one interface per family · 6 leaderboards".

Text in the concept may be imperfect; the final figure rebuilds all labels as editable text. Composition, objects, hierarchy and color are what matter.
