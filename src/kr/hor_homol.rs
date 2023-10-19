use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{ChainComplexTrait, HomologyCalc, HomologySummand};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::SpVec;
use yui_utils::bitseq::BitSeq;

use super::base::VertGen;
use super::data::KRCubeData;
use super::hor_cube::{KRHorCube, KRHorComplex};

type KRHorHomolSummand<R> = (HomologySummand<R>, Vec<LinComb<VertGen, R>>);

pub(crate) struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    v_coords: BitSeq,
    q_slice: isize,
    complex: KRHorComplex<R>,
    cache: UnsafeCell<HashMap<usize, KRHorHomolSummand<R>>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let complex = cube.as_complex();
        let cache = UnsafeCell::new( HashMap::new() );

        Self { v_coords, q_slice, complex, cache }
    }

    fn compute_summand(&self, i: usize) -> KRHorHomolSummand<R> { 
        let i = i as isize;
        let d0 = self.complex.d_matrix(i - 1);
        let d1 = self.complex.d_matrix(i);

        let h = HomologyCalc::calculate(d0, d1, true);
        let r = h.rank();

        let xs = self.complex.gens(i);
        let gens: Vec<_> = (0..r).map(|k| { 
            let v = h.gen(k); 
            let terms = v.iter().map(|(i, a)| {
                let x = xs[i].clone();
                (x, a.clone())
            });
            LinComb::from_iter(terms)
        }).collect();

        // debug_assert!(gens.iter().all(|z| !z.is_zero()));
        // debug_assert!(gens.iter().all(|z| self.complex.differentiate(i, z).is_zero()));

        (h, gens)
    }

    fn summand(&self, i: usize) -> &KRHorHomolSummand<R> {
        let cache = unsafe { &mut *self.cache.get() };
        cache.entry(i).or_insert_with(|| {
            self.compute_summand(i)
        })
    }

    pub fn rank(&self, i: usize) -> usize { 
        self.summand(i).0.rank()
    }

    pub fn gens(&self, i: usize) -> &Vec<LinComb<VertGen, R>> { 
        &self.summand(i).1
    }

    pub fn vectorize(&self, i: usize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1 == self.v_coords
        ));
        let v = self.complex.vectorize(i as isize, z);
        let v = self.summand(i).0.vectorize(&v);
        v
    }
}

#[cfg(test)]
mod tests { 
    use num_traits::Zero;
    use yui_link::Link;
    use yui_ratio::Ratio;
    use crate::kr::base::EdgeRing;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

    fn make_hml(l: &Link, v: BitSeq, q: isize) -> KRHorHomol<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Rc::new(data);
        KRHorHomol::new(rc, v, q)
    }

    #[test]
    fn rank() { 
        let l = Link::trefoil();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let hml = make_hml(&l, v, q);

        assert_eq!(hml.rank(0), 0);
        assert_eq!(hml.rank(1), 0);
        assert_eq!(hml.rank(2), 1);
        assert_eq!(hml.rank(3), 1);
    }

    #[test]
    fn gens() { 
        let l = Link::trefoil();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let hml = make_hml(&l, v, q);

        let zs = hml.gens(2);
        assert_eq!(zs.len(), 1);

        let z = &zs[0];
        let dz = hml.complex.differentiate(2, z); 
        assert!(dz.is_zero());
    }

    #[test]
    fn vectorize() { 
        let l = Link::trefoil().mirror();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 1;
        let hml = make_hml(&l, v, q);

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