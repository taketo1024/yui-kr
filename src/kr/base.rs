use derive_more::Display;
use yui_core::Elem;
use yui_lin_comb::Gen;
use yui_polynomial::{PolyN, Mono, MultiDeg};
use yui_utils::bitseq::BitSeq;

pub(crate) type MonGen = Mono<'x', MultiDeg<usize>>;
pub(crate) type EdgeRing<R> = PolyN<'x', R>;

#[derive(PartialEq, Eq, Hash, Default, Clone, Debug, Display, PartialOrd, Ord)]
#[display(fmt = "({}-{}, {})", _0, _1, _2)]
pub(crate) struct VertGen(
    pub BitSeq, // h-coords
    pub BitSeq, // v-coords
    pub MonGen
);

impl Elem for VertGen {
    fn math_symbol() -> String {
        String::from("")
    }
}
impl Gen for VertGen {}