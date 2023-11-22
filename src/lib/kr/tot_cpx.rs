use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

use cartesian::cartesian;
use delegate::delegate;
use itertools::Itertools;
use log::info;
use num_traits::Zero;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use yui::bitseq::BitSeq;
use yui::{EucRing, EucRingOps, Ring, RingOps};
use yui_homology::{isize2, ChainComplex2, GridTrait, GridIter, Grid2, ChainComplexTrait, RModStr, DisplayForGrid, rmod_str_symbol, ChainComplexCommon};
use yui_matrix::sparse::{SpVec, SpMat};

use super::base::KRChain;
use super::data::KRCubeData;
use super::tot_cube::KRTotCube;

#[derive(Default)]
pub struct KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    gens: Vec<KRChain<R>>
}

impl<R> KRTotComplexSummand<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(gens: Vec<KRChain<R>>) -> Self { 
        Self { gens }
    }

    pub fn gens(&self) -> &Vec<KRChain<R>> { 
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
            let gens = data.verts_of_weight(j).iter().flat_map(|&v| {
                cube.vert(v).gens(i)
            }).collect_vec();

            info!("  C[{idx}] : {}", gens.len());

            KRTotComplexSummand::new(gens)
        });

        Self { data, cube, summands }
    }

    #[inline(never)] // for profilability
    pub fn vectorize(&self, idx: isize2, z: &KRChain<R>) -> SpVec<R> {
        let (i, j) = (idx.0 as usize, idx.1 as usize);

        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1.weight() == j
        ));

        let z_decomp = decomp(z);

        let (r, entries) = self.data.verts_of_weight(j).iter().fold((0, vec![]), |acc, &v| { 
            let (r0, mut entries) = acc;
            let (r, vec) = self.vectorize_v(i, v, z_decomp.get(&v));

            if let Some(vec) = vec { 
                entries.extend(vec.iter().map(|(k, a)| (r0 + k, a.clone())));
            }

            (r0 + r, entries)
        });

        SpVec::from_entries(r, entries)
    }

    #[inline(never)] // for profilability
    pub fn vectorize_v(&self, i: usize, v: BitSeq, z_v: Option<&KRChain<R>>) -> (usize, Option<SpVec<R>>) {
        let h_v = self.cube.vert(v);
        let r = h_v.rank(i);
        let vec = z_v.map(|z_v| h_v.vectorize(i, &z_v));
        (r, vec)
    }


    #[inline(never)] // for profilability
    pub fn as_chain(&self, idx: isize2, v: &SpVec<R>) -> KRChain<R> { 
        let gens = self.get(idx).gens();
        v.iter().map(|(i, a)| 
            &gens[i] * a
        ).sum()
    }

    fn d_matrix_for(&self, idx: isize2) -> SpMat<R> { 
        let m = self.rank(idx + self.d_deg());
        let n = self.rank(idx);

        info!("  d-matrix {idx}, size: {:?}.", (m, n));

        let entries = if crate::config::is_multithread_enabled() { 
            (0..n).into_par_iter().map(|j| { 
                self.d_matrix_col(idx, n, j)
            }).collect::<Vec<_>>()
        } else { 
            (0..n).into_iter().map(|j| { 
                self.d_matrix_col(idx, n, j)
            }).collect()
        };

        SpMat::from_entries((m, n), entries.into_iter().flatten())
    }

    #[inline(never)] // for profilability
    fn d_matrix_col(&self, idx: isize2, n: usize, j: usize) -> Vec<(usize, usize, R)> {
        let ej = SpVec::unit(n, j);
        let z = self.as_chain(idx, &ej);        
        let w = self.d(idx, &z);
        let dj = self.vectorize(idx + self.d_deg(), &w);

        dj.iter().map(|(i, a)| 
            (i, j, a.clone())
        ).collect_vec()
    }

    pub fn reduced(self) -> ChainComplex2<R> {
        self.as_generic().reduced(false)
    }
}

fn decomp<R>(z: &KRChain<R>) -> HashMap<BitSeq, KRChain<R>>
where R: Ring, for<'x> &'x R: RingOps<R> {
    let mut res = z.iter().fold(HashMap::new(), |mut res, (v, r)| {
        res.entry(v.1).or_insert(KRChain::zero()).add_pair_ref((v, r));
        res
    });

    res.iter_mut().for_each(|(_, f)| f.clean());

    res 
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
    type Element = KRChain<R>;

    fn rank(&self, i: isize2) -> usize {
        self.summands[i].rank()
    }

    fn d_deg(&self) -> isize2 {
        isize2(0, 1)
    }

    fn d(&self, _i: isize2, z: &Self::Element) -> Self::Element {
        self.cube.d(z)
    }

    fn d_matrix(&self, idx: isize2) -> SpMat<Self::R> {
        self.d_matrix_for(idx)
    }
}


#[cfg(test)]
mod tests { 
    use yui_link::Link;
    use yui::Ratio;
    
    use yui_homology::{ChainComplexCommon, RModStr};
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