#![allow(unused)] // TODO remove
use std::sync::Arc;

use cartesian::cartesian;
use delegate::delegate;
use itertools::Itertools;
use num_traits::Zero;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use yui_core::{EucRing, EucRingOps, isize2};
use yui_homology::utils::ChainReducer;
use yui_homology::{ChainComplex2, GridTrait, GridIter, Grid2, ChainComplexTrait};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpVec, SpMat, MatType};

use super::base::VertGen;
use super::data::KRCubeData;
use super::tot_cube::KRTotCube;
use super::tot_homol::KRTotHomol;

pub struct KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    gens: Grid2<Vec<LinComb<VertGen, R>>>,
    cube: KRTotCube<R>
}

impl<R> KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        let cube = KRTotCube::new(data.clone(), q_slice);

        let n = cube.dim() as isize;
        let range = cartesian!(0..=n, 0..=n).map(|(i, j)| isize2(i, j));

        let gens = Grid2::generate(range, |idx| { 
            let (i, j) = (idx.0 as usize, idx.1 as usize);
            data.verts(j).into_iter().flat_map(|v| {
                cube.vert(v).gens(i)
            }).collect()    
        });

        Self { data, gens, cube }
    }

    fn vectorize(&self, idx: isize2, z: &LinComb<VertGen, R>) -> SpVec<R> {
        let (i, j) = (idx.0 as usize, idx.1 as usize);

        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1.weight() == j
        ));

        let (r, entries) = self.data.verts(j).iter().fold((0, vec![]), |acc, &v| { 
            let (mut r, mut entries) = acc;

            let h_v = self.cube.vert(v);
            let z_v = z.filter_gens(|x| x.1 == v); // project z to h_v.

            if !z_v.is_zero() {
                let vec = h_v.vectorize(i, &z_v);
                for (i, a) in vec.iter() { 
                    entries.push((i + r, a.clone()))
                }
            }

            r += h_v.rank(i);

            (r, entries)
        });

        SpVec::from_entries(r, entries)
    }

    fn d_matrix_for(&self, idx: isize2, q: &SpMat<R>) -> SpMat<R> { 
        let idx1 = idx + self.d_deg();

        let (m, n) = (self.rank(idx1), q.cols());
        let gens = self.get(idx);

        let entries: Vec<_> = (0..n).into_par_iter().map(|l| { 
            let v = q.col_vec(l);
            let z = v.iter().map(|(k, a)| 
                &gens[k] * a
            ).sum::<LinComb<_, _>>();
            
            let dz = z.apply(|x| self.cube.d(x));
            let w = self.vectorize(idx1, &dz);

            w.iter().map(|(k, b)| 
                (k, l, b.clone())
            ).collect_vec()
        }).flatten().collect();

        SpMat::from_entries((m, n), entries)
    }

    pub fn reduced(self) -> ChainComplex2<R> {
        let mut reducer = ChainReducer::new(self.support(), self.d_deg(), true);

        for idx in reducer.support() {
            let d = if let Some(t) = reducer.trans(idx) {
                self.d_matrix_for(idx, t.backward_mat())
            } else { 
                self.d_matrix(idx)
            };
            reducer.set_matrix(idx, d);
            reducer.reduce_at(idx);
        }

        reducer.into_complex()
    }
}

impl<R> GridTrait<isize2> for KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize2>;
    type E = Vec<LinComb<VertGen, R>>;

    delegate! { 
        to self.gens { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize2) -> bool;
            fn get(&self, i: isize2) -> &Self::E;
        }
    }
}

impl<R> ChainComplexTrait<isize2> for KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type R = R;
    type Element = LinComb<VertGen, R>;

    fn rank(&self, i: isize2) -> usize {
        self.gens[i].len()
    }

    fn d_deg(&self) -> isize2 {
        isize2(0, 1)
    }

    fn d(&self, i: isize2, z: &Self::Element) -> Self::Element {
        z.apply(|x| self.cube.d(x))
    }

    fn d_matrix(&self, idx: isize2) -> SpMat<Self::R> {
        let n = self.rank(idx);
        let q = SpMat::id(n);
        self.d_matrix_for(idx, &q)
    }
}


#[cfg(test)]
mod tests { 
    use yui_link::Link;
    use yui_ratio::Ratio;
    
    use yui_homology::{ChainComplexCommon, RModStr};
    use super::*;

    type R = Ratio<i64>;

    fn make_cpx(link: &Link, q_slice: isize) -> KRTotComplex<R> {
        let data = Arc::new( KRCubeData::<R>::new(&link, 2) );
        KRTotComplex::new(data, q_slice)
    }
    
    #[test]
    fn rank() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let c = make_cpx(&l, q);
        
        assert_eq!(c.rank(isize2(0, 0)), 0);
        assert_eq!(c.rank(isize2(0, 1)), 0);
        assert_eq!(c.rank(isize2(0, 2)), 0);
        assert_eq!(c.rank(isize2(0, 3)), 0);
        assert_eq!(c.rank(isize2(1, 0)), 0);
        assert_eq!(c.rank(isize2(1, 1)), 0);
        assert_eq!(c.rank(isize2(1, 2)), 0);
        assert_eq!(c.rank(isize2(1, 3)), 0);
        assert_eq!(c.rank(isize2(2, 0)), 1);
        assert_eq!(c.rank(isize2(2, 1)), 3);
        assert_eq!(c.rank(isize2(2, 2)), 6);
        assert_eq!(c.rank(isize2(2, 3)), 4);
        assert_eq!(c.rank(isize2(3, 0)), 1);
        assert_eq!(c.rank(isize2(3, 1)), 3);
        assert_eq!(c.rank(isize2(3, 2)), 6);
        assert_eq!(c.rank(isize2(3, 3)), 4);
    }

    #[test]
    fn vectorize() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let c = make_cpx(&l, q);
        let zs = c.get(isize2(3, 3));

        assert_eq!(zs.len(), 4);

        let _0 = R::from(0);
        let _1 = R::from(1);

        assert_eq!(c.vectorize(isize2(3, 3), &zs[0]), SpVec::from(vec![_1,_0,_0,_0]));
        assert_eq!(c.vectorize(isize2(3, 3), &zs[1]), SpVec::from(vec![_0,_1,_0,_0]));
        assert_eq!(c.vectorize(isize2(3, 3), &(&zs[0] - &zs[3])), SpVec::from(vec![_1,_0,_0,-_1]));
    }

    #[test]
    fn complex() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let c = make_cpx(&l, 0);

        c.check_d_all();

        let c = make_cpx(&l, 4);

        c.check_d_all();
    }

    #[test]
    fn homology() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let c = make_cpx(&l, q);
        let h = c.as_generic().homology(false);

        assert_eq!(h[(0, 0)].rank(), 0);
        assert_eq!(h[(0, 1)].rank(), 0);
        assert_eq!(h[(0, 2)].rank(), 0);
        assert_eq!(h[(0, 3)].rank(), 0);
        assert_eq!(h[(1, 0)].rank(), 0);
        assert_eq!(h[(1, 1)].rank(), 0);
        assert_eq!(h[(1, 2)].rank(), 0);
        assert_eq!(h[(1, 3)].rank(), 0);
        assert_eq!(h[(2, 0)].rank(), 0);
        assert_eq!(h[(2, 1)].rank(), 0);
        assert_eq!(h[(2, 2)].rank(), 0);
        assert_eq!(h[(2, 3)].rank(), 1);
        assert_eq!(h[(3, 0)].rank(), 0);
        assert_eq!(h[(3, 1)].rank(), 1);
        assert_eq!(h[(3, 2)].rank(), 0);
        assert_eq!(h[(3, 3)].rank(), 0);
    }
}