use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use delegate::delegate;
use num_traits::Zero;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, XHomologySummand, Grid1};
use yui_matrix::sparse::SpVec;
use yui::bitseq::BitSeq;

use super::base::{KRGen, KRChain, extend_ends_bounded};
use super::data::KRCubeData;
use super::hor_cpx::KRHorComplex;
use super::hor_excl::KRHorExcl;

pub struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    excl: Arc<KRHorExcl<R>>,
    inner: Grid1<XHomologySummand<KRGen, R>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize, v_coords: BitSeq) -> Self { 
        let n = data.dim() as isize;
        Self::new_restr(data, q_slice, v_coords, 0..=n)
    }

    pub fn new_restr(data: Arc<KRCubeData<R>>, q_slice: isize, v_coords: BitSeq, h_range: RangeInclusive<isize>) -> Self { 
        let n = data.dim() as isize;
        let h_range = extend_ends_bounded(h_range, 1, 0..=n);

        let excl = data.excl(v_coords);
        let complex = KRHorComplex::new_restr(data.clone(), q_slice, v_coords, h_range);
        let inner = complex.homology();
        
        Self { v_coords, q_slice, excl, inner }
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn rank(&self, i: isize) -> usize {
        self.inner[i].rank()
    }

    pub fn gens(&self, i: isize) -> Vec<KRChain<R>> { 
        let r = self.rank(i);
        (0..r).map(|k| { 
            let v = SpVec::unit(r, k);
            self.as_chain(i, &v)
        }).collect()
    }

    #[inline(never)] // for profilability
    pub fn vectorize(&self, i: isize, z: &KRChain<R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() as isize == i && 
            x.1 == self.v_coords()
        ));

        if self.rank(i).is_zero() { 
            return SpVec::zero(0)
        }

        let z_exc = self.excl.forward(z);
        let h = &self.inner[i];
        h.vectorize(&z_exc)
    }

    pub fn as_chain(&self, i: isize, v_hml: &SpVec<R>) -> KRChain<R> {
        let h = &self.inner[i];
        let z_exc = h.as_chain(v_hml);
        self.excl.backward(&z_exc, true)
    }
}

impl<R> GridTrait<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = std::vec::IntoIter<isize>;
    type Output = XHomologySummand<KRGen, R>;

    delegate! { 
        to self.inner { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize) -> bool;
            fn get(&self, i: isize) -> &Self::Output;
        }
    }
}

impl<R> Index<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = XHomologySummand<KRGen, R>;
    
    fn index(&self, index: isize) -> &Self::Output { 
        self.get(index)
    }
}

#[cfg(test)]
mod tests { 
    use yui_link::Link;
    use yui::Ratio;
    use super::*;

    type R = Ratio<i64>;

    fn make_hml(l: &Link, q: isize, v: BitSeq) -> KRHorHomol<R> {
        let data = KRCubeData::<R>::new(l);
        let rc = Arc::new(data);
        KRHorHomol::new(rc, q, v)
    }

    #[test]
    fn rank() { 
        let l = Link::trefoil();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let hml = make_hml(&l, q, v);

        assert_eq!(hml[0].rank(), 0);
        assert_eq!(hml[1].rank(), 0);
        assert_eq!(hml[2].rank(), 1);
        assert_eq!(hml[3].rank(), 1);
    }


    #[test]
    fn rank2() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from([1,0,0]);
        let q = -4;
        let hml = make_hml(&l, q, v);

        assert_eq!(hml[0].rank(), 0);
        assert_eq!(hml[1].rank(), 0);
        assert_eq!(hml[2].rank(), 0);
        assert_eq!(hml[3].rank(), 1);
    }

    #[test]
    fn vectorize() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 1;
        let hml = make_hml(&l, q, v);

        let zs = hml.gens(2);
        assert_eq!(zs.len(), 2);

        let z0 = &zs[0];
        let z1 = &zs[1];
        let w = z0 * R::from(3) + z1 * R::from(-1);

        assert_eq!(hml.vectorize(2, z0), SpVec::from(vec![R::from(1), R::from(0)]));
        assert_eq!(hml.vectorize(2, z1), SpVec::from(vec![R::from(0), R::from(1)]));
        assert_eq!(hml.vectorize(2, &w), SpVec::from(vec![R::from(3), R::from(-1)]));
    }
}