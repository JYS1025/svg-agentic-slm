# Official OpenVGLab cache to Gemma named-token training

The production `omnisvg_discrete` SFT path uses the released OpenVGLab 4B
training encoder's already-audited integer sequences. It does not invoke the
encoder while training and does not use the older inspired SVG parser.

## Pinned provenance

- OpenVGLab source revision: `812489fd9d191e39fe94bc0c4027e5d0121e0fc6`
- `configs/tokenization.yaml` SHA-256:
  `bd56bc2bb9b39f614a9d553d2319332a18a075c4239bb71ec437822b20d97dd4`
- SQLite cache SHA-256:
  `da25dc3db8739e846d952b3044e5f7977eaf8820f4ea0c45c46ed8111f91a3fa`
- Audit manifest SHA-256:
  `30e3deb8a8cf443a1c32a58bc48e41b35b4adc615c14eb7fb9dee1f48d4e35c7`
- Train input SHA-256:
  `7920c33235ede566657003ac63166b01b3e594b6ccc521d34d9fbe937e5642bd`
- Validation input SHA-256:
  `ab4a24667b19160a3c3337ad9b25a8e1c4c4ba58116ec690c3829d25b621ef2a`
- Test input SHA-256:
  `381d28d516273698a1c4e0b793376fe90aa7135ad89b69a04ec6cbda57ab9d3b`

All hashes, split counts, cache metadata, source lines, and sample IDs must
match. The cache does not contain an independent SVG hash. SVG identity is
therefore proven by the pinned whole-file input hash plus split, source line,
and unique sample ID; the loader records the observed SVG SHA-256 for traceability.

## Namespace and capability boundary

The Gemma tokenizer receives the ordered named namespace
`<svgovg4b:151938>` through `<svgovg4b:196999>`, 45,062 tokens total. This is
the conservative contiguous source interval that covers every ID producible by
the pinned 4B training configuration, including BOS `196998` and EOS `196999`.
The integer is provenance, not a request to reuse Qwen embeddings. Exact
one-token registration creates new Gemma rows, and only those tied rows plus
LoRA parameters are trained.

This dialect is cache-only and encode-only. No Qwen checkpoint compatibility,
official decoder, generation grammar, or round trip is claimed. Inference must
fail closed until a separately audited decoder is implemented. The older
`omnisvg-inspired-discrete-svg` dialect remains available only through an
explicit legacy/toy opt-in.
