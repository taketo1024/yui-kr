use std::ops::Add;
use derive_more::Display;
use num_traits::Zero;
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TripGrad(
    pub isize, 
    pub isize, 
    pub isize
);

impl Add for TripGrad {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        TripGrad(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Add<(isize, isize, isize)> for TripGrad {
    type Output = Self;
    fn add(self, rhs: (isize, isize, isize)) -> Self::Output {
        TripGrad(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Zero for TripGrad {
    fn zero() -> Self {
        Self(0, 0, 0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0 && self.1 == 0 && self.2 == 0
    }
}