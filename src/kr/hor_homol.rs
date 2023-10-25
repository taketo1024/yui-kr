use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;
use delegate::delegate;

use yui_core::{EucRing, EucRingOps};
use yui_homology::{ReducedComplex, PrintSeq, Graded};
use yui_homology::{ChainComplexTrait, HomologySummand, utils::HomologyCalc};
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
    reduced: ReducedComplex<R>,
    cache: UnsafeCell<HashMap<usize, KRHorHomolSummand<R>>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let complex = cube.as_complex();
        let reduced = complex.inner().reduced(true);
        let cache = UnsafeCell::new( HashMap::new() );

        Self { v_coords, q_slice, complex, reduced, cache }
    }

    fn compute_summand(&self, i: usize) -> KRHorHomolSummand<R> { 
        let i = i as isize;
        let d0 = self.reduced.d_matrix(i - 1);
        let d1 = self.reduced.d_matrix(i);

        let h = HomologyCalc::calculate(d0, d1, true);
        let r = h.rank();

        let xs = self.complex.gens(i);
        let gens: Vec<_> = (0..r).map(|k| { 
            let v = h.gen(k);                           // vec for reduced.
            let w = self.reduced.trans_backward(i, &v); // vec for original.

            let terms = w.iter().map(|(i, a)| {
                let x = &xs[i];
                (x.clone(), a.clone())
            });
            LinComb::from_iter(terms)
        }).collect();

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

        let h = &self.summand(i).0;
        let i = i as isize;

        let v = self.complex.vectorize(i, z);      // vec for complex 
        let v = self.reduced.trans_forward(i, &v); // vec for reduced
        let v = h.trans_forward(&v);               // vec for homology

        v
    }

    pub fn print_complex(&self) { 
        self.complex.print_seq()
    }

    pub fn print_reduced(&self) { 
        self.reduced.print_seq()
    }
}

impl<R> Graded<isize> for KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = std::vec::IntoIter<isize>;

    delegate! { 
        to self.complex { 
            fn support(&self) -> Self::Itr;
        }
    }

    fn display(&self, i: isize) -> String {
        self.summand(i as usize).0.to_string()
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
    fn rank2() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from_iter([1,0,0]);
        let q = -4;
        let hml = make_hml(&l, v, q);

        assert_eq!(hml.rank(0), 0);
        assert_eq!(hml.rank(1), 0);
        assert_eq!(hml.rank(2), 0);
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
        let dz = hml.complex.d(2, z); 
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