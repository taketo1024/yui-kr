# Feedback on yui 1.0.0, from porting yui-kr

*Written while porting `yui-kr` from its May 2026 state (`f1aa017`) to
yui `update-1.0.0` (`09bc4134`, `yui-core` 0.5.0). These are the places
where the new API felt awkward from the downstream side. Taketo's
decisions are recorded per item.*

## 1. `BitSeq::generate(n)` no longer compiles bare

`BitSeq` became generic, `BitSeq<I: BitRepr = u64>`
(`lib-core/src/conc/misc/bitseq.rs:130`). The default applies in *type*
position but not to associated-function calls, so

```rust
BitSeq::generate(n).filter(|v| ...).count()   // ambiguous: `I` unconstrained
```

no longer resolves whenever the result isn't otherwise constrained. The
workaround is `<BitSeq>::generate(n)`, which is correct but obscure —
few readers know that `<T>` in type position applies defaults while
`T::` does not.

This bit exactly once in `yui-kr` (`tot_cube.rs:40`), because every other
call site had its `I` pinned by a surrounding `BitSeq`-typed value. A
crate that uses `generate` more would hit it repeatedly.

**Suggestion:** a `pub type BitSeq64 = BitSeq<u64>;` alias, or inherent
constructors on the `u64` case.

**Status: accepted** — "good idea".

## 2. `as_u64` → `as_u128` forces a cast back

The only numeric accessor on `BitSeq` is now
(`lib-core/src/conc/misc/bitseq.rs:171`):

```rust
pub fn as_u128(&self) -> u128 {
    self.val.to_u128()        // = self.val as u128
}
```

`self.val` has type `I`, so for `BitSeq<u64>` the value is widened
`u64 → u128`, and the caller narrows it straight back:

```rust
pub fn vert(&self, v_coords: BitSeq) -> &KRHorHomol<R> {
    let i = v_coords.as_u128() as usize;   // u64 → u128 → usize
    &self.verts[i]
}
```

Two conversions to read back a word that was already `u64`. This costs
nothing at runtime — it compiles away — so it is an ergonomics and
safety point, not performance. Two separate concerns:

**(a) The return type doesn't track `I`.** `BitSeq<u8>` also hands you a
`u128`. There is no way to get the value at its natural width. A
`pub fn val(&self) -> I` returning the backing word as-is would cover
that, with `as_u128()` kept for the width-erased case.

**(b) The dominant use is indexing, and the cast is unchecked.** Packing
a cube vertex into an array index is exactly what `vert` does. `as usize`
is a silently truncating cast: for `BitSeq<u128>` with `len > 64` it
wraps and returns the *wrong vertex* — no panic, no debug assert. An
`as_usize()` doing `try_into().unwrap()` would turn that into a loud
failure and remove the cast from every caller. (`yui-kr` is not at risk
today — its sequences are `n`-crossing-sized — but the sharp edge is in
the API.)

There is also a small readability wrinkle: `Bit::as_u64()` still exists
(`bitseq.rs:35`) and returns `0`/`1` as a `u64`. So in one module, the
type holding a *single bit* has the `u64` accessor, and the type holding
up to 128 of them has the `u128` one. Not a bug, just momentarily
confusing when reading.

If `BitSeq64` lands for item 1, `as_usize()` would pair naturally with it.

## 3. `check_d_all` behind `test-utils` is a silent trap

`ChainComplex::check_d_all` (and `check_d_at`, `check_d_for`) are now
gated by `#[cfg(any(test, feature = "test-utils"))]`
(`lib-homology/src/conc/complex.rs:244–265`).

From downstream this vanishes from `#[cfg(test)]` code with a plain
"method not found in `ChainComplex<...>`" and no hint that a feature flag
exists. It cost 11 call sites in `yui-kr` and a detour into yui's source
to diagnose. The fix is one line:

```toml
[dev-dependencies]
yui-homology = { path = "../yui/lib-homology", features = ["test-utils"] }
```

but it is easy to miss precisely because `yui-link` already had a
`test-utils` entry, so the block *looked* correct.

**Suggestion:** a line in the release notes; the compiler cannot help here.

**Status: under consideration** — making it public is being reconsidered.

## 4. `Slot` is a real improvement

`lib-link`'s replacement of raw `usize` node indices with the `Slot` enum
is a clear win downstream. In `yui-kr`'s `collect_x_polys`:

```rust
let (a,b,c) = if Self::is_vert(x, signs[i]) {
    (3,2,0)                              // before
    (Slot::NW, Slot::NE, Slot::SW)       // after
} else {
    (2,1,3)
    (Slot::NE, Slot::SE, Slot::NW)
};
```

The `NodeType::paired_slot` rewrite `H => i ^ 1` is also equivalent to
the old `(5 - slot) % 4` on all four inputs, and reads better.

**Status: no change needed** — "good!".

## 5. `Link::test_data` chirality is a silent behaviour change

Not an API complaint, but the highest-cost item of the whole port.

Commit `e0a06738` replaced the hard-coded PD codes with KnotInfo's own
diagrams (previously KnotAtlas), mirroring `3_1, 5_1, 5_2, 6_1, 6_2,
7_1, 7_2, 7_3, 8_19`. `4_1`, `6_3` and the links are unchanged.

This breaks **no compile**. It silently makes downstream numbers wrong:
all 30 of `yui-kr`'s test failures traced to it, via `writhe` flipping
`-3 → +3` for `3_1`. Diagnosing it meant deriving `w` back out of a
grading mismatch (`isize3(-2,4,2)` vs `isize3(4,-2,-4)`).

**Suggestion:** call this out prominently in the release notes, with the
list of affected knots. Any downstream crate with hard-coded expectations
against `test_data` needs the same mirror treatment.

---

## Appendix: gaps in the downstream migration notes

The hand-written breakage list I was given was accurate on what it
covered, but incomplete. What it missed, in rough order of cost:

| Missing item | Cost |
|---|---|
| `Ring`/`RingOps`/`EucRing`/`EucRingOps`/`AddMon`/`MathType` moved to `yui_core::abst` (commit `b2e11215`) | 13 files — the bulk of the port |
| `TeX` moved to `yui_core::util::tex` | 1 file |
| `check_d_all` gated behind `test-utils` | 11 call sites |
| `SyncCounter::new()` → `new(n)` | 1 site |
| `BitSeq` generic: `as_u64` → `as_u128`, `generate` inference | 2 sites |
| `Link::test_data` chirality | 30 test failures |

Two items in the list were also slightly off: `Sign`/`GetSign` was
recorded as "no direct use in yui-kr" but is used in both `base.rs` and
`data.rs`; and the `Slot` migration in `lib-link` was not mentioned at
all, though it predates the 1.0.0 merge rather than coming from it.

Things the list got exactly right and saved time on: the `tex` feature
removal, `Lc::clean` / `add_pair_ref`, the `total_rank` false alarm, the
`make_matrix` false alarm, and the dependency-major bumps.
