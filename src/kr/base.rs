use std::iter::zip;

use derive_more::{Display, DebugCustom};
use itertools::{Itertools, FoldWhile};
use yui::{Elem, Sign, PowMod2, GetSign};
use yui::lc::Gen;
use yui::poly::{PolyN, MultiVar};
use yui::bitseq::{BitSeq, Bit};

pub(crate) type BaseMono = MultiVar<'x', usize>;
pub(crate) type BasePoly<R> = PolyN<'x', R>;

#[derive(PartialEq, Eq, Hash, Default, Clone, Display, DebugCustom, PartialOrd, Ord)]
#[display(fmt = "({}-{}, {})", _0, _1, _2)]
#[debug(fmt = "{}", self)]
pub struct VertGen(
    pub BitSeq, // h-coords
    pub BitSeq, // v-coords
    pub BaseMono
);

impl Elem for VertGen {
    fn math_symbol() -> String {
        String::from("")
    }
}

impl Gen for VertGen {}

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

    (-1).pow_mod2(e).sign()
}

#[cfg(test)]
mod tests { 
    use super::*;

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


}