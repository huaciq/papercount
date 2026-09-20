# Regular-first hybrid detector

The current track pipeline keeps the negative-gradient, per-strip, curved-track
and geometry sequence stages.  It now estimates a continuous local regularity
score `R(y)` from robust spacing consistency, local profile autocorrelation,
track quality/density, and a gap-geometry proxy.  The components are normalized
weights from `default.yaml` and the score is interpolated at every sequence
transition.  Samples are labelled `REGULAR`, `MIXED`, or `IRREGULAR` only for
diagnostics; the objective changes continuously rather than at a hard boundary.

In regular regions the adaptive spacing prior increases the cost of an
unusually large gap and widens the evidence search corridor.  A gap near an
integer multiple of the local spacing creates a search hypothesis only.  A
candidate must still be present in the existing pool or in multiple negative
strip profiles before it can enter the sequence.  No image evidence means no
new paper, and unresolved hypotheses are recorded in `unresolved_regular_gaps`.

In mixed and irregular regions the gap profile, fitted strip-wise curves,
non-crossing checks, and asymmetric short/long penalties dominate.  Long gaps
are therefore acceptable when the geometry is genuinely irregular.  Triple
competition records the A-B, B-C and A-C alternatives so a short parallel
secondary texture can be suppressed without hard-deleting a close real sheet.

The detector runs a conservative pass 1 over strong tracks, computes the map
once, then runs pass 2 only in anomaly intervals (large gaps, short-gap
competition, marginal candidates, suppressed strong tracks, and boundary
margin).  Regularity is not repeatedly re-estimated from recovered candidates.

RAW masking is now edge-connected background suppression.  A row becomes
`INVALID` only when brightness, P90, raw texture, run length, and border
connectivity all agree.  Internal dark paper seams remain usable.  Strong
tracks may expand the boundary mask once via
`refine_stack_mask_with_strong_tracks`; marginal or period-only proposals may
not do so.  Margin rows are usable with higher direct evidence requirements.

`regularity_map.png`, `hybrid_sequence.png`, `raw_stack_mask.png`, and
`debug_info.json` expose the score components, region ratios, recovery type,
mask refinement, competition records, and unresolved gaps.  An optional
`ground_truth.json` can provide `{ "image.bmp": {"count": 90,
"edge_y": [...]}}`; `--ground-truth` reports count error and, when edges are
present, matched/false-positive/false-negative precision, recall and F1.
