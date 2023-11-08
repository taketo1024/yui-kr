use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use yui_core::{EucRing, EucRingOps};
use yui_homology::{Homology, HomologySummand, GridTrait, RModStr, GridIter, Grid, XModStr};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::SpVec;
use yui_utils::bitseq::BitSeq;

use super::base::VertGen;
use super::data::KRCubeData;
use super::hor_cpx::KRHorComplex;
use super::hor_excl::KRHorExcl;

pub struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    excl: KRHorExcl<R>,
    gens: Grid<XModStr<VertGen, R>>,
    homology: Homology<R>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let complex = KRHorComplex::new(data, v_coords, q_slice);
        let homology = complex.homology();
        let (excl, gens) = complex.take_data();
        Self { v_coords, q_slice, excl, gens, homology }
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn rank(&self, i: usize) -> usize {
        self.homology[i as isize].rank()
    }

    pub fn homol_gens(&self, i: usize) -> Vec<LinComb<VertGen, R>> { 
        let h = &self.homology[i as isize];
        let r = h.rank();

        (0..r).map(|k| { 
            let v = SpVec::unit(r, k);
            let z = self.as_chain(i, &v);
            z
        }).collect()
    }

    pub fn vectorize(&self, i: usize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1 == self.v_coords()
        ));

        let i = i as isize;
        let h = &self.homology[i];
        let t = h.trans().unwrap();

        let z_exc = self.excl.forward(z);
        let v_exc = self.gens[i].vectorize(&z_exc);
        let v_hml = t.forward(&v_exc);
        
        v_hml
    }

    pub fn as_chain(&self, i: usize, v_hml: &SpVec<R>) -> LinComb<VertGen, R> {
        let i = i as isize;
        let h = &self.homology[i];
        let t = h.trans().unwrap();

        let v_exc = t.backward(&v_hml);
        let z_exc = self.gens[i].as_chain(&v_exc);
        let z = self.excl.backward(&z_exc, true);

        z
    }
}

impl<R> GridTrait<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize>;
    type E = HomologySummand<R>;

    delegate! { 
        to self.homology {
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize) -> bool;
            fn get(&self, i: isize) -> &Self::E;
        }
    }
}

impl<R> Index<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = HomologySummand<R>;
    delegate! { 
        to self.homology { 
            fn index(&self, index: isize) -> &Self::Output;
        }
    }
}

#[cfg(test)]
mod tests { 
    use yui_link::Link;
    use yui_ratio::Ratio;
    use super::*;

    type R = Ratio<i64>;

    fn make_hml(l: &Link, v: BitSeq, q: isize) -> KRHorHomol<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Arc::new(data);
        KRHorHomol::new(rc, v, q)
    }

    #[test]
    fn rank() { 
        let l = Link::trefoil();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let hml = make_hml(&l, v, q);

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
        let hml = make_hml(&l, v, q);

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
        let hml = make_hml(&l, v, q);

        let zs = hml.homol_gens(2);
        assert_eq!(zs.len(), 2);

        let z0 = &zs[0];
        let z1 = &zs[1];
        let w = z0 * R::from(3) + z1 * R::from(-1);

        assert_eq!(hml.vectorize(2, z0), SpVec::from(vec![R::from(1), R::from(0)]));
        assert_eq!(hml.vectorize(2, z1), SpVec::from(vec![R::from(0), R::from(1)]));
        assert_eq!(hml.vectorize(2, &w), SpVec::from(vec![R::from(3), R::from(-1)]));
    }
}