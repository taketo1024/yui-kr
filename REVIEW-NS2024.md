# Review: yui-kr vs NS2024 (arXiv:2111.00388v2)

*Code review comparing the implementation against the paper
"Computations of HOMFLY homology" (Nakagane–Sano, 2024).
Reviewed at commit `7cf974f`, yui at `09bc4134` (`update-1.0.0`).
Date: 2026-08-06.*

**Verdict: the implementation is mathematically correct.**
Every piece of the algorithm traces to the paper, and no correctness error
was found on the field-coefficient path that the paper and all published
results use. No issue below changes any published result.

*Follow-up: the issues were triaged by Taketo and addressed on branch
`strengthen-tests`. A/B are by design and were downgraded, C was reworded
after correction, D/E/F are fixed. Resolution is recorded per item.*

## What was verified, item by item

| Paper | Code | How verified |
|---|---|---|
| (2.1) global shift `{−w+b−1, w+b−1, w−b+1}` | `data.rs:152` `l_grad_shift` | symbol-by-symbol |
| Fig 1 grading shifts (all 8 cells, both signs) | `data.rs:156` `x_grad_shift` | cell-by-cell |
| Fig 1 maps: hor `X_k−X_i`, `(X_k−X_i)(X_l−X_i)`; ver `X_l−X_i`, `1` | `data.rs:259`, `data.rs:272` | derived `x_ac = X_k−X_i`, `x_bc = X_l−X_i` from the slot diagrams, then matched all 4 sign/bit cases each |
| Lemma 3.2 / Remark 3.3 (edge ring, spanning tree, `y_c` shortcut) | `data.rs:289` `collect_x_polys`, `data.rs:346` `resolve` | the traversal telescopes `X_to − X_from` correctly; tree crossings get single-variable `x_bc`, resolved crossings single-variable `x_ac` — exactly the Remark 3.3 optimization |
| §3.2 q-slicing | `hor_cube.rs:41` `mon_deg` | re-derived independently from `to_inner_grad`; then hand-recomputed the `no_red` test ranks (3, 26, 51, 28) from monomial counts — exact match |
| §3.3 exclusion, φ/ψ of [Ras15, Lemma 3.8] | `hor_excl.rs` `forward`/`backward` | re-derived the chain-map identity `D∘ψ = ψ∘d″` by hand in the code's Koszul-sign conventions, in two 2-direction configurations (including the nontrivial sign case) — the code's `ψ(z) = ((d′(ιz)+ι(d″z))/p, z)` is exactly right |
| exclusion bookkeeping | `is_free`/`remain_vars`/`rem` | later divisors cannot reintroduce excluded variables (divisors are confined to `remain_vars`); `div_rem` terminates on a well-founded order; homogeneity is preserved, so the q-grading survives |
| vertical complex `φ_{v'} ∘ d_V ∘ ψ_v` | `tot_cube.rs:101` + `tot_cpx.rs:197` | representative chains via `backward`, cube-level `d_V`, projection via `forward` + `vectorize` — the paper's composite verbatim |
| Props 2.3 / 2.6 support bounds | `data.rs:180–195` | formulas match; the `step_by(2)` ranges land in the right parity class mod `root_grad` (difference is always even) |
| boundary handling of restricted ranges | `extend_ends_bounded` in `hor_homol.rs:37`, `tot_homol.rs:53` | complexes extended ±1 in the differential direction, homology queried only in the interior; `h`-restriction safe because `d_deg = (0,1)` doesn't couple h-columns |
| Theorem 3 (HOMFLY Euler char) | `result.rs:111` `homfly_poly` | `(−1)^{(k−j)/2} qⁱaʲ` exactly, via `t ↦ −1`; plus the 14 passing `check_homfly` tests against KnotInfo data |
| §4 output format (Δ-slices, [CG24]) | `result.rs:146` `delta_table` | matches; the `b3_1` test *is* the paper's page-9 table |
| `size_limit` partial computation (§4, Fig 3) | `calc.rs:66` + `tot_cpx_red.rs` | traced the fake-zero propagation: skipped differentials keep their endpoints' ranks above the limit, so the three-rank guard provably blocks recording homology anywhere a faked matrix could leak in; skipped entries stay `?` |

## Independent recomputation performed

Beyond the existing test suite:

- `3₁`/`m3₁` and `5₂`/`m5₂` computed from scratch in both chiralities.
  Both pairs satisfy Prop 2.5 (`ℋ^{i,j,k}(mK) = ℋ^{−i,−j,−k}(K)`) exactly.
  The two runs of a pair build genuinely different complexes (different
  signs, shifts, edge polys, exclusions), so this cross-validates the
  entire sign-convention stack end to end. The `5₂` table also matches the
  paper's §5 "correct homology" table digit for digit.
- Batch fixtures: k8 (21 passed) and k9 (48 passed, 1 ignored) in release.
  The ≤9-crossing range is anchored to Rasmussen's independent
  computations, so this is external validation, not self-agreement.
- `check_d_matrices` now runs in release builds and never fired across any
  of these runs.

## Issues found

### A. Non-unit lead coefficient would panic over `ℤ` — `hor_excl.rs:132`

`find_excl_var_in` only *prefers* `±1` coefficients (`max_by` on
`is_pm_one`), but `div_rem` at `hor_excl.rs:408` *asserts*
`a0.is_unit()`. Over `ℚ` every nonzero coefficient is a unit, so this
never fires. Over `ℤ`, a slice whose only candidate term is `2x_k` would
panic.

**Resolution: by design, no change.** The code is verified for `ℚ` only.
It is written generically for readability and future work; supporting `ℤ`
requires building the theory first, not patching the exclusion step.
Recorded here as a known limitation of the `ℤ` path.

### B. Torsion dropped over `ℤ` — `tot_cpx.rs:42`

`KRTotComplexSummand::tors()` returns `[]` unconditionally, but the chain
groups of the vertical complex are horizontal *homologies*, which can carry
`ℤ`-torsion. Over a field this is correct, since there is no torsion.

**Resolution: by design, no change**, for the same reason as A. My original
suggestion to bound the stage by `Field` was wrong — it would give up the
genericity deliberately kept for future work.

### C. `ℋ(D)` is only an invariant for braid closures — `kr.rs:26`

*(Corrected: my original heading said the theory "rejects" such diagrams.
That is wrong. `C(D)` and `ℋ(D)` are defined for **any** link diagram; it is
the **invariance** that requires `D` to be a braid closure. The §5 example
shows a minimal-crossing diagram of `5₂` giving a different — not
ill-defined — answer.)*

`from_link` takes any diagram, and nothing recorded this.

**Resolution: fixed.** Doc comment on `from_link` citing Remark 2.2 and the
`5₂` example. No behaviour change — the entry point stays permissive, which
is correct, since computing `ℋ(D)` for a general `D` is legitimate.

### D. `Index<isize3>` fallback aliased invalid gradings — `kr.rs:81`

`to_inner_grad(idx).unwrap_or(isize3(0,0,0))` sent gradings with no valid
inner index to inner `(0,0,0)`.

*(Reclassified: **this was not reachable.** `Grid::index` returns a default
for absent keys, and the maximum inner `q` over the support is*
`q_max = (s − 1 − 3n)/2`. *The Seifert graph of a connected diagram has `s`
vertices and `n` edges, so `s − 1 ≤ n` and hence `q_max ≤ −n < 0` for every
`n ≥ 1`: the fallback always missed `q_range` and returned rank 0, confirmed
by probing 15 knots. The sole case with `q_max = 0` is the crossingless
unknot, rejected earlier. So this was latent fragility resting on an
accidental inequality, not a live bug.)*

**Resolution: fixed.** `KRHomology` now holds a `zero` summand and `index`
matches on `to_inner_grad`, returning it when there is no inner grading.
Regression test `kr::tests::index_out_of_support`.

### E. Prop 2.4 was assumed, invisibly

`to_inner_grad` folds `i > 0` onto `(−i, j, k+2i)` (`data.rs:234`),
`is_supported` prunes by `k_range.contains(k+2i)` (`data.rs:229`), and
`KRHomologyStr::set` writes the reflected index (`result.rs:75`). All three
are theorem-backed and halve the computation, but none said so, and nothing
checked the assumption.

**Resolution: fixed.** `// note:` comments citing Prop 2.4 at all three
sites, plus `kr::tests::prop_2_4_independent`, which rebuilds the `i > 0`
half from its own inner grading and compares. Not circular: the unfolded
cell always differs from the folded one, and for at least one target the
`q`-slice lies outside the computed `q_range`, so it was never built.

### F. Confusing failure on 0-crossing diagrams — `data.rs`

Found while probing D. `Link::unknot()` died inside `collect_x_polys` at
`assert_eq!(resolved.n_comps(), 1)` with `left: 0, right: 1`.

**Resolution: fixed.** Explicit
`assert!(link.n_crossings() > 0, "0-crossing diagrams are not supported.")`
in `new_no_excl`, beside the existing `is_knot` check.

## Smaller improvement suggestions

- **Document the inner↔outer grading map.**
  `i = 2q + 2h + i₀, j = 2h + j₀, k = 2v + k₀` at `data.rs:250` takes real
  derivation to connect to `mon_deg`; three lines of comment (and "Fig 1"
  citations at `x_grad_shift`/`hor_edge_poly`) would save the next reader
  an afternoon.
- The traversal panic arms at `data.rs:304` encode the PD-code convention
  (under-strand always enters at SW, etc.). A one-line `// note:` would
  turn a mysterious `panic!()` into a checkable invariant.
- The empty-iterator hack at `hor_cube.rs:56` deserves a cleaner API on
  `KRMono::generate` (yui-side).
- `trans_for` / the `make_matrix` import in `hor_excl.rs:396` are used
  only by tests now — fine to keep, but worth knowing it's test-only API.
- k10/k11 fixtures exist but were not run in this review (k8/k9 were).
  A full `--features batch_test` run before the release would close the
  loop on the whole published table.

## Scope caveat

`ψ`'s chain-map identity was verified by hand in two representative
configurations plus the passing hand-computed unit tests. yui-homology's
`ChainReducer`/`Summand` machinery was treated as trusted (it has its own
suite, and the level-0/1/2 agreement tests exercise it here). Everything
else above was checked directly.

## Follow-up: tests added

Branch `strengthen-tests` turns the hand-verifications above into
regression tests. The lib suite goes from 58 to 82.

| File | Tests |
|---|---|
| `data.rs` | `x_grad_shift_fig1` (all 8 Figure 1 cells), `edge_polys_fig1` (all 4 `(sign, bit)` cases, horizontal and vertical), `grad_shift_positive`, `root_grad_positive_trefoil`, `grad_roundtrip`, `support_bounds` (Props 2.3/2.6 + parity), `range_of_values` |
| `base.rs` | `edge_sign_square` (every 2-face anti-commutes — the `d² = 0` condition), `combine_decombine`, `extend_ends` |
| `hor_excl.rs` | `forward_backward_id` (`φψ = id` per [Ras15, Lemma 3.8], on every surviving generator of every vertex), `div_rem_linear`, `div_rem_quadratic` |
| `tot_cpx.rs` | `decomp_partitions` |
| `result.rs` | `set_maintains_pairing`, `get_semantics`, `mirror_involution`, `delta_and_thin`, `homfly_poly_trefoil` (Thm 3 — previously only checked by the `#[ignore]`d integration test) |
| `kr.rs` | `prop_2_4_independent`, `mirror_duality_3_1`, `mirror_duality_5_2`, `amphichiral_4_1`, `index_out_of_support` |

Deliberately **not** added: a Prop 2.4 assertion on `structure()` output.
The folding in `to_inner_grad` bakes the symmetry in, so such a test would
pass by construction; `prop_2_4_independent` is the honest version.

The new tests were checked to actually bite, by mutation:

- flipping one sign in `x_grad_shift` fails 13 tests, `x_grad_shift_fig1`
  and `root_grad_positive_trefoil` among them;
- breaking the parity count in `sign_between` fails 29, including
  `edge_sign_square`, `forward_backward_id` and `prop_2_4_independent`.

Verified green afterwards: 82 lib + 14 `k3-7` tests in debug and release,
`k8` (21) and `k9` (48) batch fixtures, and clippy unchanged at its 14
pre-existing warnings.
