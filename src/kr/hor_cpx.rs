use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use yui_core::{Ring, RingOps, EucRing, EucRingOps};
use yui_homology::{ReducedComplex, Grid, ChainComplexTrait, XModStr, GridTrait, GridIter, ChainComplexSummand, Homology, make_matrix, ChainComplex};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpVec, SpMat};
use yui_utils::bitseq::BitSeq;

use crate::kr::hor_cube::KRHorCube;

use super::base::VertGen;
use super::data::KRCubeData;
use super::hor_excl::KRHorExcl;

pub struct KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    excl: KRHorExcl<R>,
    gens: Grid<XModStr<VertGen, R>>,
    inner: ReducedComplex<R>
} 

impl<R> KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let excl = KRHorExcl::from(&data, v_coords, 2);
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let gens = Self::excl_gens(&excl, &cube);
        let inner = Self::make_cpx(&excl, &gens).reduced(true);

        Self { v_coords, q_slice, excl, gens, inner }
    }

    fn excl_gens(excl: &KRHorExcl<R>, cube: &KRHorCube<R>) -> Grid<XModStr<VertGen, R>> {
        let n = cube.data().dim() as isize;
        Grid::new(0..=n, |i| {
            let xs = cube.gens(i as usize);
            let red_xs = excl.reduce_gens(&xs);
            XModStr::from_iter(red_xs)
        })
    }

    fn make_cpx(excl: &KRHorExcl<R>, gens: &Grid<XModStr<VertGen, R>>) -> ChainComplex<R> {
        ChainComplex::new(
            gens.support(), 1, 
            |i| make_matrix(gens[i].gens(), gens[i+1].gens(), |x| excl.diff_red(x))
        )
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn vectorize(&self, i: isize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        let z_exc = self.excl.forward(z);
        let v_exc = self.gens[i].vectorize(&z_exc);
        let v = self.inner.trans(i).forward(&v_exc);
        v
    }

    pub fn as_chain(&self, i: isize, v: &SpVec<R>) -> LinComb<VertGen, R> {
        let v_exc = self.inner.trans(i).backward(v);
        let z_exc = self.gens[i].as_chain(&v_exc);
        let z = self.excl.backward(&z_exc);
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

#[cfg(test)]
mod tests { 
    use yui_homology::RModStr;
    use yui_link::Link;
    use yui_ratio::Ratio;

    use super::*;

    type R = Ratio<i64>;

    fn make_cpx(link: &Link, v_coords: BitSeq, q_slice: isize, level: usize, red: bool) -> KRHorComplex<R> {
        let data = Arc::new( KRCubeData::<R>::new(&link) );

        let excl = KRHorExcl::from(&data, v_coords, level);
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let gens = KRHorComplex::excl_gens(&excl, &cube);
        let cpx = KRHorComplex::make_cpx(&excl, &gens);
        let inner = if red { 
            cpx.reduced(true)
        } else { 
            ReducedComplex::bypass(&cpx)
        };

        KRHorComplex { v_coords, q_slice, excl, gens, inner }
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