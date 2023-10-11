use std::ops::{RangeInclusive, Index};
use std::rc::Rc;
use yui_core::{Ring, RingOps};
use yui_homology::FreeChainComplex;
use yui_utils::bitseq::BitSeq;
use crate::kr::hor_cube::KRHorCube;

use super::base::{VertGen};
use super::data::KRCubeData;

pub(crate) struct KRHorHomol<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    complex: FreeChainComplex<VertGen, R, RangeInclusive<isize>>
} 

impl<R> KRHorHomol<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let complex = cube.as_complex();
        Self { complex }
    }
}

impl<R> Index<usize> for KRHorHomol<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Output = ();

    fn index(&self, _index: usize) -> &Self::Output {
        todo!()
    }
}