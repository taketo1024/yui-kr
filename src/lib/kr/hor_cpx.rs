use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use yui::{Ring, RingOps, EucRing, EucRingOps};
use yui_homology::{ChainComplexTrait, GridTrait, GridIter, XChainComplex, XChainComplexSummand, XHomology, Grid1, XModStr};
use yui_matrix::sparse::SpMat;
use yui::bitseq::BitSeq;

use crate::kr::hor_cube::KRHorCube;

use super::base::{KRGen, KRChain};
use super::data::KRCubeData;
use super::hor_excl::KRHorExcl;

pub struct KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    q_slice: isize,
    v_coords: BitSeq,
    excl: Arc<KRHorExcl<R>>,
    inner: XChainComplex<KRGen, R>
} 

impl<R> KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize, v_coords: BitSeq) -> Self { 
        let excl = data.excl(v_coords);
        let cube = KRHorCube::new(data.clone(), q_slice, v_coords);
        let inner = Self::make_cpx(excl.clone(), cube);

        Self { v_coords, q_slice, excl, inner }
    }

    fn make_cpx(excl: Arc<KRHorExcl<R>>, cube: KRHorCube<R>) -> XChainComplex<KRGen, R> { 
        let n = cube.dim() as isize;
        let summands = Grid1::generate(0..=n, |i| { 
            let gens = cube.gens(i as usize).filter(|v| 
                excl.should_remain(v)
            );
            XModStr::free(gens)
        });
        
        XChainComplex::new(summands, 1, move |_, e| {
            excl.d(e)
        })
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }
}

impl<R> GridTrait<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Itr = GridIter<isize>;
    type Output = XChainComplexSummand<KRGen, R>;

    delegate! { 
        to self.inner { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize) -> bool;
            fn get(&self, i: isize) -> &Self::Output;
        }
    }
}

impl<R> Index<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Output = XChainComplexSummand<KRGen, R>;

    fn index(&self, i: isize) -> &Self::Output {
        self.get(i)
    }
}


impl<R> ChainComplexTrait<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type R = R;
    type Element = KRChain<R>;

    delegate! { 
        to self.inner { 
            fn rank(&self, i: isize) -> usize;
            fn d_deg(&self) -> isize;
            fn d_matrix(&self, i: isize) -> SpMat<Self::R>;
        }
    }

    fn d(&self, i: isize, z: &KRChain<R>) -> KRChain<R> { 
        let z_exc = self.excl.forward(z);
        let dz_exc = self.inner.d(i, &z_exc);
        let dz = self.excl.backward(&dz_exc, true);
        dz
    }
}

impl<R> KRHorComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn homology(&self) -> XHomology<KRGen, R> { 
        self.inner.reduced().homology(true)
    }
}

#[cfg(test)]
mod tests { 
    use yui_homology::{RModStr, ChainComplexCommon};
    use yui_link::Link;
    use yui::Ratio;

    use super::*;

    type R = Ratio<i64>;

    fn make_cpx(link: &Link, v_coords: BitSeq, q_slice: isize, level: usize, red: bool) -> KRHorComplex<R> {
        let data = Arc::new( KRCubeData::<R>::new_excl(link, level) );
        let excl = data.excl(v_coords);
        let cube = KRHorCube::new(data.clone(), q_slice, v_coords);
        let inner = KRHorComplex::make_cpx(excl.clone(), cube);
        let inner = if red { 
            inner.reduced()
        } else { 
            inner
        };

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