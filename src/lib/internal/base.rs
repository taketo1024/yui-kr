//! The generators and chains every stage is written in terms of.
//!
//! A generator [`KRGen`] is a cube vertex `(h, v)` together with a monomial in
//! the edge ring; a chain is a linear combination of these. [`combine`] and
//! [`decombine`] move between an `R`-chain and the polynomial-coefficient form
//! `KRPolyChain`, in which the cube differentials are easiest to apply.

use std::iter::zip;
use std::ops::RangeInclusive;

use derive_more::{Display, Debug};
use itertools::{Itertools, FoldWhile};
use num_traits::One;
use yui_core::abst::{MathType, Ring, RingOps};
use yui_core::num::Sign;
use yui_core::lc::{LcKey, Lc};
use yui_core::poly::{PolyN, MultiVar, MonoOrd};
use yui_core::bitseq::{BitSeq, Bit};

pub(crate) type KRMono = MultiVar<'x', usize>;
pub(crate) type KRPoly<R> = PolyN<'x', R>;

pub type KRChain<R> = Lc<KRGen, R>;
pub type KRPolyChain<R> = Lc<KRGen, KRPoly<R>>;

#[derive(PartialEq, Eq, Hash, Default, Clone, Display, Debug)]
#[display("<{}; {}-{}>", _2, _0, _1)]
#[debug("{}", self)]
pub struct KRGen(
    pub BitSeq, // h-coords
    pub BitSeq, // v-coords
    pub KRMono
);

impl MathType for KRGen {
    fn math_symbol() -> String {
        String::from("")
    }
}

impl PartialOrd for KRGen {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl Ord for KRGen {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // order by priority: v > h > mono
        BitSeq::cmp(&self.1, &other.1).then_with(||
            BitSeq::cmp(&self.0, &other.0)
        ).then_with(||
            KRMono::cmp_grlex(&self.2, &other.2)
        )
    }
}

impl LcKey for KRGen {}

pub(crate) fn sign_between(from: BitSeq, to: BitSeq) -> Sign { 
    use Bit::*;
    
    assert_eq!(from.len(), to.len());
    debug_assert_eq!(to.weight() - from.weight(), 1);
    
    let e = zip(from.iter(), to.iter()).fold_while(0, |c, (f, t)| 
        match (f, t) {
            (Bit0, Bit0) => FoldWhile::Continue(c),
            (Bit1, Bit1) => FoldWhile::Continue(c + 1),
            (Bit0, Bit1) => FoldWhile::Done(c),
            (Bit1, Bit0) => panic!()
        }
    ).into_inner() as u32;

    Sign::from_parity(e)
}

pub fn combine<R>(z: KRChain<R>) -> KRPolyChain<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    z.into_iter().map(|(v, r)| {
        let w = KRGen(v.0, v.1, KRMono::one());
        let p = KRPoly::from((v.2, r));
        (w, p)
    }).collect()
}

pub fn decombine<R>(z: KRPolyChain<R>) -> KRChain<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    debug_assert!(z.iter().all(|(v, _)| v.2.is_one()));

    z.into_iter().flat_map(|(v, p)| { 
        p.into_iter().map(move |(x, a)| {
            let v = KRGen(v.0, v.1, x);
            (v, a)
        })
    }).collect()
}

pub(crate) fn extend_ends_bounded(r: RangeInclusive<isize>, d: isize, bound: RangeInclusive<isize>) -> RangeInclusive<isize> { 
    let start = isize::max(r.start() - d, *bound.start());
    let end   = isize::min(r.end()   + d, *bound.end());
    start..=end
}

#[cfg(test)]
mod tests {
    use yui_core::num::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn edge_sign() {
        use Sign::*;
        
        let e = sign_between(
            BitSeq::from([0,0,0]), 
            BitSeq::from([1,0,0]));
        assert_eq!(e, Pos);

        let e = sign_between(
            BitSeq::from([0,0,0]), 
            BitSeq::from([0,1,0]));
        assert_eq!(e, Pos);

        let e = sign_between(
            BitSeq::from([1,0,0]), 
            BitSeq::from([1,1,0]));
        assert_eq!(e, Neg);

        let e = sign_between(
            BitSeq::from([0,1,0]),
            BitSeq::from([1,1,0]));
        assert_eq!(e, Pos);
    }

    // Every 2-face of the cube must anti-commute, which is what makes d² = 0.
    #[test]
    fn edge_sign_square() {
        let n = 4;

        for u in <BitSeq>::generate(n) {
            let zeros = (0..n).filter(|&i| u[i].is_zero()).collect_vec();

            for [&i, &j] in zeros.iter().array_combinations() {
                let ui  = u.edit(|b| b.set_1(i));
                let uj  = u.edit(|b| b.set_1(j));
                let uij = u.edit(|b| { b.set_1(i); b.set_1(j) });

                let via_i = sign_between(u, ui) * sign_between(ui, uij);
                let via_j = sign_between(u, uj) * sign_between(uj, uij);

                assert_eq!(via_i, -via_j, "at {u:?}, ({i}, {j})");
            }
        }
    }

    #[test]
    fn combine_decombine() {
        let h = BitSeq::from([1,0,0]);
        let v = BitSeq::from([0,1,0]);
        let x = |i, d| KRMono::from((i, d));

        // two monomials sharing (h, v), one on a different h.
        let z = KRChain::<R>::from_iter([
            (KRGen(h, v, x(0, 1)),  R::from(2)),
            (KRGen(h, v, x(1, 2)), -R::from(3)),
            (KRGen(v, v, x(0, 1)),  R::from(5)),
        ]);

        assert_eq!(decombine(combine(z.clone())), z);

        // combine merges the monomials sharing (h, v) into one polynomial.
        let c = combine(z);
        assert_eq!(c.nterms(), 2);
        assert!(c.iter().all(|(w, _)| w.2.is_one()));
    }

    #[test]
    fn extend_ends() {
        assert_eq!(extend_ends_bounded(2..=3, 1, 0..=5), 1..=4);
        assert_eq!(extend_ends_bounded(0..=2, 2, 0..=5), 0..=4); // clamped below
        assert_eq!(extend_ends_bounded(3..=5, 2, 0..=5), 1..=5); // clamped above
        assert_eq!(extend_ends_bounded(0..=5, 1, 0..=5), 0..=5); // clamped both
        assert_eq!(extend_ends_bounded(2..=2, 0, 0..=5), 2..=2); // no extension
    }
}