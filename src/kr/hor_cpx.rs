use std::rc::Rc;
use delegate::delegate;

use yui_core::{Ring, RingOps, EucRing, EucRingOps};
use yui_homology::{ReducedComplex, Grid, ChainComplexTrait, XModStr, GridTrait, GridIter, ChainComplexSummand, Homology};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpVec, SpMat};
use yui_utils::bitseq::BitSeq;

use crate::kr::hor_cube::KRHorCube;

use super::base::VertGen;
use super::data::KRCubeData;

pub(crate) struct KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    gens: Grid<XModStr<VertGen, R>>,
    inner: ReducedComplex<R>
} 

impl<R> KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let n = data.dim() as isize;
        let range = 0..=n;

        let cube = KRHorCube::new(data, v_coords, q_slice);
        let gens = Grid::new(range.clone(), |i|
            XModStr::from_iter( cube.gens(i as usize) )
        );

        // TODO use exclusion.
        let inner = cube.as_complex().reduced(true);

        Self { v_coords, q_slice, gens, inner }
    }

    pub fn vectorize(&self, i: isize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        let v = self.gens[i].vectorize(z);
        let v = self.inner.trans(i).forward(&v);
        v
    }

    pub fn as_chain(&self, i: isize, v: &SpVec<R>) -> LinComb<VertGen, R> {
        let v = self.inner.trans(i).backward(v);
        let z = self.gens[i].as_chain(&v);
        z
    }

    pub fn d(&self, i: isize, z: &LinComb<VertGen, R>) -> LinComb<VertGen, R> { 
        let v = self.vectorize(i, z);
        let w = self.inner.d(i, &v);
        let dz = self.as_chain(i + 1, &w);
        dz
    }
}

impl<R> GridTrait<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Itr = GridIter<isize>;
    type E = ChainComplexSummand<R>;

    delegate! { 
        to self.inner { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize) -> bool;
            fn get(&self, i: isize) -> &Self::E;
        }
    }
}

impl<R> ChainComplexTrait<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type R = R;

    delegate! { 
        to self.inner { 
            fn d_deg(&self) -> isize;
            fn d_matrix(&self, i: isize) -> &SpMat<Self::R>;
        }
    }
}

impl<R> KRHorComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn homology(&self) -> Homology<R> { 
        self.inner.homology(true)
    }
}
