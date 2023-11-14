#![allow(unused)] use std::ops::Index;
// TODO remove
use std::sync::Arc;

use cartesian::cartesian;
use delegate::delegate;
use itertools::Itertools;
use num_traits::Zero;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use yui::{EucRing, EucRingOps, isize2, Ring, RingOps};
use yui_homology::utils::ChainReducer;
use yui_homology::{ChainComplex2, GridTrait, GridIter, Grid2, ChainComplexTrait, RModStr, DisplayForGrid, rmod_str_symbol};
use yui::lc::LinComb;
use yui_matrix::sparse::{SpVec, SpMat, MatType};

use super::base::VertGen;
use super::data::KRCubeData;
use super::tot_cube::KRTotCube;
use super::tot_homol::KRTotHomol;

#[derive(Default)]
pub struct KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    gens: Vec<LinComb<VertGen, R>>
}

impl<R> KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(gens: Vec<LinComb<VertGen, R>>) -> Self { 
        Self { gens }
    }

    pub fn gens(&self) -> &Vec<LinComb<VertGen, R>> { 
        &self.gens
    }
}

impl<R> RModStr for KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type R = R;

    fn rank(&self) -> usize {
        self.gens.len()
    }

    fn tors(&self) -> &[R] {
        &[]
    }
}

impl<R> DisplayForGrid for KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn display_for_grid(&self) -> String {
        rmod_str_symbol(self.rank(), self.tors(), ".")
    }
}

pub struct KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    cube: KRTotCube<R>,
    summands: Grid2<KRTotComplexSummand<R>>
}

impl<R> KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        let cube = KRTotCube::new(data.clone(), q_slice);

        let n = cube.dim() as isize;
        let range = cartesian!(0..=n, 0..=n).map(|(i, j)| isize2(i, j));

        let summands = Grid2::generate(range, |idx| { 
            let (i, j) = (idx.0 as usize, idx.1 as usize);
            let gens = data.verts(j).into_iter().flat_map(|v| {
                cube.vert(v).gens(i)
            }).collect();
            KRTotComplexSummand::new(gens)
        });

        Self { data, cube, summands }
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
        let gens = self.get(idx).gens();

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
        }).collect();

        SpMat::from_entries((m, n), entries.into_iter().flatten())
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
    type Output = KRTotComplexSummand<R>;

    delegate! { 
        to self.summands { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize2) -> bool;
            fn get(&self, i: isize2) -> &Self::Output;
        }
    }
}

impl<R> Index<(isize, isize)> for KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = KRTotComplexSummand<R>;

    fn index(&self, index: (isize, isize)) -> &Self::Output {
        self.get(index.into())
    }
}

impl<R> ChainComplexTrait<isize2> for KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type R = R;
    type Element = LinComb<VertGen, R>;

    fn rank(&self, i: isize2) -> usize {
        self.summands[i].rank()
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
    use yui::Ratio;
    
    use yui_homology::{ChainComplexCommon, RModStr, DisplayTable};
    use super::*;

    type R = Ratio<i64>;

    fn make_cpx(link: &Link, q_slice: isize) -> KRTotComplex<R> {
        let data = Arc::new( KRCubeData::<R>::new(link, 2) );
        KRTotComplex::new(data, q_slice)
    }
    
    #[test]
    fn rank() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let c = make_cpx(&l, q);
        
        assert_eq!(c[(0, 0)].rank(), 0);
        assert_eq!(c[(0, 1)].rank(), 0);
        assert_eq!(c[(0, 2)].rank(), 0);
        assert_eq!(c[(0, 3)].rank(), 0);
        assert_eq!(c[(1, 0)].rank(), 0);
        assert_eq!(c[(1, 1)].rank(), 0);
        assert_eq!(c[(1, 2)].rank(), 0);
        assert_eq!(c[(1, 3)].rank(), 0);
        assert_eq!(c[(2, 0)].rank(), 1);
        assert_eq!(c[(2, 1)].rank(), 3);
        assert_eq!(c[(2, 2)].rank(), 6);
        assert_eq!(c[(2, 3)].rank(), 4);
        assert_eq!(c[(3, 0)].rank(), 1);
        assert_eq!(c[(3, 1)].rank(), 3);
        assert_eq!(c[(3, 2)].rank(), 6);
        assert_eq!(c[(3, 3)].rank(), 4);
    }

    #[test]
    fn vectorize() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let c = make_cpx(&l, q);

        let gens = c[(3, 3)].gens();

        assert_eq!(gens.len(), 4);

        let r0 = R::from(0);
        let r1 = R::from(1);

        assert_eq!(c.vectorize(isize2(3, 3), &gens[0]), SpVec::from(vec![r1, r0, r0, r0]));
        assert_eq!(c.vectorize(isize2(3, 3), &gens[1]), SpVec::from(vec![r0, r1, r0, r0]));
        assert_eq!(c.vectorize(isize2(3, 3), &(&gens[0] - &gens[3])), SpVec::from(vec![r1, r0, r0, -r1]));
    }

    #[test]
    fn complex() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let c = make_cpx(&l, 0);

        c.check_d_all();

        let c = make_cpx(&l, -4);

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


    #[test]
    fn complex_red() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let c = make_cpx(&l, q).reduced();

        assert_eq!(c[(0, 0)].rank(), 0);
        assert_eq!(c[(0, 1)].rank(), 0);
        assert_eq!(c[(0, 2)].rank(), 0);
        assert_eq!(c[(0, 3)].rank(), 0);
        assert_eq!(c[(1, 0)].rank(), 0);
        assert_eq!(c[(1, 1)].rank(), 0);
        assert_eq!(c[(1, 2)].rank(), 0);
        assert_eq!(c[(1, 3)].rank(), 0);
        assert_eq!(c[(2, 0)].rank(), 0);
        assert_eq!(c[(2, 1)].rank(), 0);
        assert_eq!(c[(2, 2)].rank(), 0);
        assert_eq!(c[(2, 3)].rank(), 1);
        assert_eq!(c[(3, 0)].rank(), 0);
        assert_eq!(c[(3, 1)].rank(), 1);
        assert_eq!(c[(3, 2)].rank(), 0);
        assert_eq!(c[(3, 3)].rank(), 0);
    }
}