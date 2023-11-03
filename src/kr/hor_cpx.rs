use std::ops::Index;
use std::rc::Rc;
use delegate::delegate;

use yui_core::{Ring, RingOps, EucRing, EucRingOps};
use yui_homology::{ReducedComplex, Grid, ChainComplexTrait, XModStr, GridTrait, GridIter, ChainComplexSummand, Homology, make_matrix};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpVec, SpMat};
use yui_utils::bitseq::BitSeq;

use crate::kr::hor_cube::KRHorCube;

use super::base::VertGen;
use super::data::KRCubeData;
use super::hor_excl::KRHorExcl;

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

        let excl = KRHorExcl::from(&data, v_coords, 2);
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let gens = Grid::new(0..=n, |i|
            XModStr::from_iter(cube.gens(i as usize) )
        );

        let inner = Self::excl_cpx(&excl, &gens);
        let inner = inner.reduced(true);

        Self { v_coords, q_slice, gens, inner }
    }

    fn excl_cpx(excl: &KRHorExcl<R>, gens: &Grid<XModStr<VertGen, R>>) -> ReducedComplex<R> {
        let red_gens = Grid::new(
            gens.support(), 
            |i| excl.reduce_gens( &gens[i].gens() )
        );

        ReducedComplex::new(
            gens.support(), 1, 
            |i| make_matrix(&red_gens[i], &red_gens[i + 1], |v| excl.diff_red(v)),
            |i| excl.trans_for(&gens[i].gens(), &red_gens[i])
        )
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

impl<R> Index<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Output = ChainComplexSummand<R>;

    fn index(&self, i: isize) -> &Self::Output {
        self.get(i)
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

#[allow(unused_imports)] // TODO remove later.
#[cfg(test)]
mod tests { 
    use yui_homology::{RModStr, DisplaySeq};
    use yui_link::Link;
    use yui_ratio::Ratio;

    use crate::kr::base::BasePoly;

    use super::*;

    type R = Ratio<i64>;
    type P = BasePoly<R>;

    fn make_cpx(link: &Link, v: BitSeq, q: isize, l: usize, red: bool) -> KRHorComplex<R> {
        let data = Rc::new( KRCubeData::<R>::new(&link) );
        let n = data.dim() as isize;

        let excl = KRHorExcl::from(&data, v, l);
        let cube = KRHorCube::new(data, v, q);
        let gens = Grid::new(0..=n, |i|
            XModStr::from_iter(cube.gens(i as usize) )
        );

        let mut inner = KRHorComplex::excl_cpx(&excl, &gens);
        if red { 
            inner = inner.reduced(true);
        }

        KRHorComplex { v_coords: v, q_slice: q, gens, inner }
    }

    #[test]
    fn no_red() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 0, false);

        assert_eq!(c[0].rank(), 3);
        assert_eq!(c[1].rank(), 26);
        assert_eq!(c[2].rank(), 51);
        assert_eq!(c[3].rank(), 28);

        c.check_d_all();

        let h = c.inner.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level1() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 1, false);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 3);
        assert_eq!(c[2].rank(), 10);
        assert_eq!(c[3].rank(), 7);

        c.check_d_all();

        let h = c.inner.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level2() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 2, false);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.check_d_all();

        let h = c.inner.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level1_red() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 1, true);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.check_d_all();

        let h = c.inner.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level2_red() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 2, true);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.check_d_all();

        let h = c.inner.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }
}