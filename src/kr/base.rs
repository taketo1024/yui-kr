use std::ops::Add;

use num_traits::Zero;
use yui_polynomial::{PolyN, Mono, MDegree};
use yui_homology::FreeRModStr;

pub(crate) type MonGen = Mono<'x', MDegree<usize>>;
pub(crate) type VertMod<R> = FreeRModStr<MonGen, R>;
pub(crate) type EdgeRing<R> = PolyN<'x', R>;

#[derive(PartialEq, Eq, Debug)]
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