use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use yui_core::{EucRing, EucRingOps};
use yui_homology::{Homology, HomologySummand, GridTrait, RModStr, GridIter, Grid};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::SpVec;
use yui_utils::bitseq::BitSeq;

use super::base::VertGen;
use super::data::KRCubeData;
use super::hor_cpx::KRHorComplex;

pub struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    complex: KRHorComplex<R>,
    homology: Homology<R>,
    h_gens: Grid<Vec<LinComb<VertGen, R>>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let complex = KRHorComplex::new(data, v_coords, q_slice);
        let homology = complex.homology();

        let h_gens = Grid::new(homology.support(), |i| { 
            let h = &homology[i];
            let r = h.rank();

            (0..r).map(|k| { 
                let v = h.gen_vec(k).unwrap();
                let z = complex.as_chain(i, &v, true);
                z
            }).collect()
        });

        Self { complex, homology, h_gens }
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.complex.v_coords()
    }

    pub fn q_slice(&self) -> isize { 
        self.complex.q_slice()
    }

    pub fn rank(&self, i: usize) -> usize {
        self.homology[i as isize].rank()
    }

    pub fn gens(&self, i: usize) -> &Vec<LinComb<VertGen, R>> { 
        &self.h_gens[i as isize]
    }

    pub fn vectorize(&self, i: usize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1 == self.v_coords()
        ));

        let i = i as isize;

        let v = self.complex.vectorize(i, z);
        let v = self.homology[i].vec_from_cpx(&v).unwrap();
        v
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
    use num_traits::Zero;
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
    fn gens() { 
        let l = Link::trefoil();
        let v = BitSeq::from([1,0,0]);
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
        let v = BitSeq::from([1,0,0]);
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