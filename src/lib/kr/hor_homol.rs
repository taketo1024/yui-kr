use std::ops::Index;
use std::sync::Arc;

use log::info;
use num_traits::Zero;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, XHomologySummand, DisplaySeq};
use yui_matrix::sparse::SpVec;
use yui::bitseq::BitSeq;

use super::base::{KRGen, KRChain};
use super::data::KRCubeData;
use super::hor_cpx::KRHorComplex;
use super::hor_excl::KRHorExcl;

pub struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    excl: Arc<KRHorExcl<R>>,
    summands: Vec<XHomologySummand<KRGen, R>>,
    _zero: XHomologySummand<KRGen, R>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize, v_coords: BitSeq) -> Self { 
        let excl = data.excl(v_coords);
        let complex = KRHorComplex::new(data.clone(), q_slice, v_coords);

        info!("compute hor-homology, q: {q_slice}, v: {v_coords}.");
        let homol = complex.homology();
        info!("hor-homology, q: {q_slice}, v: {v_coords}\n{}", homol.display_seq());

        // extract for fast access.
        let summands = homol.into_iter().map(|(_, e)| e).collect();
        let _zero = XHomologySummand::zero();
        
        Self { v_coords, q_slice, excl, summands, _zero }
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn rank(&self, i: usize) -> usize {
        self.summands[i].rank()
    }

    pub fn gens(&self, i: usize) -> Vec<KRChain<R>> { 
        let r = self.rank(i);
        (0..r).map(|k| { 
            let v = SpVec::unit(r, k);
            self.as_chain(i, &v)
        }).collect()
    }

    #[inline(never)] // for profilability
    pub fn vectorize(&self, i: usize, z: &KRChain<R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1 == self.v_coords()
        ));

        if self.rank(i).is_zero() { 
            return SpVec::zero(0)
        }

        let z_exc = self.excl.forward(z);
        let h = &self.summands[i];
        h.vectorize(&z_exc)
    }

    pub fn as_chain(&self, i: usize, v_hml: &SpVec<R>) -> KRChain<R> {
        let h = &self.summands[i];
        let z_exc = h.as_chain(v_hml);
        self.excl.backward(&z_exc, true)
    }
}

impl<R> GridTrait<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = std::ops::Range<isize>;
    type Output = XHomologySummand<KRGen, R>;

    fn support(&self) -> Self::Itr { 
        0..self.summands.len() as isize
    }

    fn is_supported(&self, i: isize) -> bool {
        0 <= i && i < self.summands.len() as isize
    }

    fn get(&self, i: isize) -> &Self::Output { 
        if self.is_supported(i) {
            &self.summands[i as usize]
        } else { 
            &self._zero
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
        let data = KRCubeData::<R>::new(l, 2);
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