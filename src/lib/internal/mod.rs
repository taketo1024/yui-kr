//! The stages of the computation, in the order they run.
//!
//! For a knot diagram `D` with `n` crossings, `C(D) = ⊗_c C(D_c)` is a double
//! complex over the edge ring `R(D) = ℚ[x_1, ..., x_n]`, graded by `q` and by
//! the two cube coordinates `h, v ∈ {0,1}ⁿ`, with differentials `d_H` and
//! `d_V` [NS2024, §2]. Since `C(D)` is infinitely generated over `ℚ`, it is
//! sliced by the `q`-grading and each slice computed on its own [§3.2]:
//!
//! ```text
//!   data        the diagram: signs, edge polynomials, gradings, support
//!     |
//!     |   for each q-slice, and each vertex v of the vertical cube:
//!     v
//!   hor_cube    the horizontal cube: generators and d_H at fixed (q, v)
//!   hor_excl    exclusion of variables -- shrinks it before any linear algebra
//!   hor_cpx     the two combined, as a chain complex
//!   hor_homol   its homology H(D, v)
//!     |
//!     |   then, over all v:
//!     v
//!   tot_cube    the vertical cube, with vertices H(D, v) and induced d_V
//!   tot_cpx     the vertical complex
//!   tot_cpx_red its reduction, with an optional size cap
//!   tot_homol   its homology -- H'(D), which after a grading shift is H(D)
//! ```
//!
//! [`base`] holds the generators and chains shared by every stage.
//!
//! These modules are public so that tests and downstream experiments can drive
//! the stages individually. They are not a stable API — use
//! [`KRHomology`](crate::KRHomology) or [`KRCalc`](crate::KRCalc) instead.

pub mod base;
pub mod data;
pub mod hor_cube;
pub mod hor_cpx;
pub mod hor_excl;
pub mod hor_homol;
pub mod tot_cube;
pub mod tot_cpx;
pub mod tot_cpx_red;
pub mod tot_homol;
