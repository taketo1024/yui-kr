use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::rc::Rc;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{FreeChainComplex, ChainComplex, RModStr};
use yui_homology::utils::HomologyCalc;
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpMat, SpVec};
use yui_utils::bitseq::BitSeq;
use crate::kr::hor_cube::KRHorCube;

use super::base::VertGen;
use super::data::KRCubeData;

type KRHorHomolSummand<R> = (Vec<LinComb<VertGen, R>>, SpMat<R>);

pub(crate) struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    v_coords: BitSeq,
    complex: FreeChainComplex<VertGen, R, RangeInclusive<isize>>,
    cache: HashMap<usize, OnceCell<KRHorHomolSummand<R>>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let n = data.dim();
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let complex = cube.as_complex();
        let cache = (0..=n).map(|i| 
            (i, OnceCell::new())
        ).collect();

        Self { v_coords, complex, cache }
    }

    fn compute_summand(&self, i: usize) -> KRHorHomolSummand<R> { 
        let i = i as isize;
        let d0 = self.complex.d_matrix(i - 1);
        let d1 = self.complex.d_matrix(i);

        // p: (r, n), q: (n, r)
        let (h, p, q) = HomologyCalc::calculate_with_trans(d0, d1);
        let r = h.rank();

        let gens = self.complex[i].generators();
        let h_gens = (0..r).map(|k| { 
            let v = q.col_vec(k); 
            let terms = v.iter().map(|(i, a)| {
                let x = &gens[i];
                (x.clone(), a.clone())
            });
            LinComb::from_iter(terms)
        }).collect();

        (h_gens, p)
    }

    fn summand(&self, i: usize) -> &KRHorHomolSummand<R> {
        &self.cache[&i].get_or_init(|| { 
            self.compute_summand(i)
        })
    }

    pub fn rank(&self, i: usize) -> usize { 
        self.summand(i).0.len()
    }

    pub fn gens(&self, i: usize) -> &Vec<LinComb<VertGen, R>> { 
        &self.summand(i).0
    }

    pub fn trans(&self, i: usize) -> &SpMat<R> { 
        &self.summand(i).1
    }

    pub fn vectorize(&self, i: usize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1 == self.v_coords
        ));
        let v = self.complex[i as isize].vectorize(z);
        let p = self.trans(i);
        p * v
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
        let dz = hml.complex.differetiate(z); 
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