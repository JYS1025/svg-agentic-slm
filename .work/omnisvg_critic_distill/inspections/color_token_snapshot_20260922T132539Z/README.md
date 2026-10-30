# OmniSVG color-token audit

Snapshot: 2026-09-22T13:25:35.465337+00:00
Generated outputs: 1117 (failed statuses excluded)
No color token + blank render: 246 (22.02%)
All blank candidate renders: 270 (24.17%)

`missing_color/`: no model color token; upstream decoder assigns fill=none.
`has_color_visible/`: one or more color tokens and a nonblank candidate PNG.
Each example contains the exact generated SVG, rendered PNG, generated token IDs,
selected text instruction, and metadata. Source artifacts are copied unchanged.

## missing_color

- [01_index_00600/icon:1f525a7f723f3723333a9b6cfab14faf](missing_color/01_index_00600/candidate.svg) — color tokens: 0; [PNG](missing_color/01_index_00600/candidate.png)
- [02_index_01877/icon:724d7639dec431f10cad433eeb493e11](missing_color/02_index_01877/candidate.svg) — color tokens: 0; [PNG](missing_color/02_index_01877/candidate.png)
- [03_index_02846/icon:c48fc27eac17d16a43ff43662165a754](missing_color/03_index_02846/candidate.svg) — color tokens: 0; [PNG](missing_color/03_index_02846/candidate.png)
- [04_index_03958/icon:c1ce7c19709397207aaaf3a3d46efaf7](missing_color/04_index_03958/candidate.svg) — color tokens: 0; [PNG](missing_color/04_index_03958/candidate.png)
- [05_index_04993/icon:f18788b4dcfe195f80d86465ef9d7696](missing_color/05_index_04993/candidate.svg) — color tokens: 0; [PNG](missing_color/05_index_04993/candidate.png)

## has_color_visible

- [01_index_00187/icon:2262678cd5732906ebede34771788eab](has_color_visible/01_index_00187/candidate.svg) — color tokens: 2; [PNG](has_color_visible/01_index_00187/candidate.png)
- [02_index_01755/icon:0dbf0e8f1a25f1cf7b45649702de185a](has_color_visible/02_index_01755/candidate.svg) — color tokens: 35; [PNG](has_color_visible/02_index_01755/candidate.png)
- [03_index_02859/icon:70132e4215b3a044e8f7769a8d6bde08](has_color_visible/03_index_02859/candidate.svg) — color tokens: 2; [PNG](has_color_visible/03_index_02859/candidate.png)
- [04_index_03823/icon:ccffbd8415b2f08925e5e429862b0609](has_color_visible/04_index_03823/candidate.svg) — color tokens: 21; [PNG](has_color_visible/04_index_03823/candidate.png)
- [05_index_04999/icon:755ad317513fe974cf94e82f94bc1854](has_color_visible/05_index_04999/candidate.svg) — color tokens: 10; [PNG](has_color_visible/05_index_04999/candidate.png)

