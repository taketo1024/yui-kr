use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use yui_core::{Ring, RingOps, EucRing, EucRingOps};
use yui_homology::{Grid1, ChainComplexTrait, GridTrait, GridIter, XChainComplex, XChainComplexSummand, XHomology};
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
    inner: XChainComplex<VertGen, R>
} 

impl<R> KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let excl = KRHorExcl::from(&data, v_coords, 2);
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let inner = Self::make_cpx(cube, &excl, true);

        Self { v_coords, q_slice, excl, inner }
    }

    fn excl_gens(cube: &KRHorCube<R>, excl: &KRHorExcl<R>) -> Grid1<Vec<VertGen>> {
        let n = cube.data().dim() as isize;
        Grid1::generate(0..=n, |i| {
            let xs = cube.gens(i as usize);
            let red_xs = excl.reduce_gens(&xs);
            red_xs
        })
    }

    fn make_cpx(cube: KRHorCube<R>, excl: &KRHorExcl<R>, ch_red: bool) -> XChainComplex<VertGen, R> {
        let excl = excl.clone();
        let gens = Self::excl_gens(&cube, &excl);

        let cpx = XChainComplex::generate(
            gens.support(), 1, 
            move |i| gens[i].clone(),
            move |_, x| excl.diff_red(x)
        );

        if ch_red { 
            cpx.reduced()
        } else { 
            cpx
        }
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn vectorize(&self, i: isize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        let z_exc = self.excl.forward(z);
        let v_exc = self.inner[i].vectorize(&z_exc);
        v_exc
    }

    pub fn as_chain(&self, i: isize, v_exc: &SpVec<R>) -> LinComb<VertGen, R> {
        let z_exc = self.inner[i].as_chain(&v_exc);
        let z = self.excl.backward(&z_exc, false);
        z
    }

    pub fn d(&self, i: isize, z: &LinComb<VertGen, R>) -> LinComb<VertGen, R> { 
        let z_exc = self.excl.forward(z);
        let dz_exc = self.inner.d(i, &z_exc);
        let dz = self.excl.backward(&dz_exc, true);
        dz
    }

    pub(crate) fn take_data(self) -> KRHorExcl<R> { 
        self.excl
    }
}

impl<R> GridTrait<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Itr = GridIter<isize>;
    type E = XChainComplexSummand<VertGen, R>;

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
    type Output = XChainComplexSummand<VertGen, R>;

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
            fn d_matrix(&self, i: isize) -> SpMat<Self::R>;
        }
    }
}

impl<R> KRHorComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn homology(&self) -> XHomology<VertGen, R> { 
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
        let inner = KRHorComplex::make_cpx(cube, &excl, red);

        KRHorComplex { v_coords, q_slice, excl, inner }
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