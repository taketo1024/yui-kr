use std::collections::HashMap;
use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use cartesian::cartesian;
use delegate::delegate;
use log::info;
use num_traits::Zero;
use rayon::prelude::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use yui::bitseq::BitSeq;
use yui::{EucRing, EucRingOps, Ring, RingOps, AddMon};
use yui_homology::{isize2, GridTrait, GridIter, Grid2, ChainComplexTrait, ChainComplex, RModStr, DisplayForGrid, rmod_str_symbol, DisplaySeq, DisplayTable};
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

    fn zero() -> Self { 
        Self::new(vec![])
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
    q: isize,
    range: (RangeInclusive<isize>, RangeInclusive<isize>),
    cube: KRTotCube<R>,
    summands: Grid2<KRTotComplexSummand<R>>
}

impl<R> KRTotComplex<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, q: isize) -> Self { 
        let n = data.dim() as isize;
        Self::new_restr(data, q, (0..=n, 0..=n))
    }

    pub fn new_restr(data: Arc<KRCubeData<R>>, q: isize, range: (RangeInclusive<isize>, RangeInclusive<isize>)) -> Self { 
        info!("C_tot (q: {}, range: {:?})..", q, range);
        info!("setup tot-cube..");

        let cube = KRTotCube::new_restr(
            data.clone(), 
            q, 
            range.clone()
        );
        
        let support = cartesian!(
            range.0.clone(), 
            range.1.clone()
        ).map(isize2::from);

        info!("setup summands..");

        let summands = Grid2::generate_with_default(
            support, 
            |idx| Self::summand(&data, &cube, idx),
            KRTotComplexSummand::zero()
        );

        info!("C_tot (q: {}, range: {:?})\n{}", q, range, summands.display_table("h", "v"));

        Self { q, range, data, cube, summands }
    }

    fn summand(data: &KRCubeData<R>, cube: &KRTotCube<R>, idx: isize2) -> KRTotComplexSummand<R> { 
        let (i, j) = idx.into();
        let gens = data.verts_of_weight(j as usize).par_iter().flat_map(|&v| {
            cube.vert(v).gens(i)
        }).collect();
        KRTotComplexSummand::new(gens)
    }

    pub fn q_deg(&self) -> isize { 
        self.q
    }

    pub fn h_slice(&self, i: isize) -> ChainComplex<R> { 
        let v_range = self.range.1.clone();

        info!("C_tot/h (q: {}, h: {i}, v: {:?})..", self.q, v_range);
        info!("prepare matrices..");

        let c = ChainComplex::generate(v_range, 1, |j| {
            let n = self.rank((i, j).into());
            let m = self.rank((i, j + 1).into());
            
            info!("d_tot ({}, {}) -> ({}, {}), size: {:?}", 
                i, j, 
                i, j + 1, 
                (m, n)
            );

            self.d_matrix(isize2(i, j))
        });

        info!("C_tot/h (q: {}, h: {i})\n{}", self.q, c.display_seq("v"));

        c
    }

    #[inline(never)] // for profilability
    pub fn vectorize(&self, idx: isize2, z: &KRChain<R>) -> SpVec<R> {
        if self.get(idx).rank() == 0 {
            return SpVec::zero(0)
        }

        let (i, j) = idx.into();

        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() as isize == i && 
            x.1.weight() as isize == j
        ));

        let z_decomp = decomp(z);
        let zero = KRChain::zero();

        let vs = self.data.verts_of_weight(j as usize).par_iter().map(|&v| {
            let z_v = z_decomp.get(&v).unwrap_or(&zero);
            self.vectorize_v(i, v, z_v)
        }).collect::<Vec<_>>();

        SpVec::stack_vecs(vs)
    }

    #[inline(never)] // for profilability
    pub fn vectorize_v(&self, i: isize, v: BitSeq, z_v: &KRChain<R>) -> SpVec<R> {
        let h_v = self.cube.vert(v);
        h_v.vectorize(i, &z_v)
    }

    #[inline(never)] // for profilability
    pub fn as_chain(&self, idx: isize2, v: &SpVec<R>) -> KRChain<R> { 
        let gens = self.get(idx).gens();
        KRChain::sum(v.iter().map(|(i, a)| 
            &gens[i] * a
        ))
    }

    fn d_matrix_for(&self, from: isize2) -> SpMat<R> { 
        let to = from + self.d_deg();
        let m = self.rank(to);
        let n = self.rank(from);

        if m == 0 || n == 0 { 
            return SpMat::zero((m, n))
        }
        
        let cols = (0..n).into_par_iter().map(|j| { 
            self.d_matrix_col(from, n, j)
        }).collect::<Vec<_>>();

        SpMat::from_col_vecs(m, cols)
    }

    #[inline(never)] // for profilability
    fn d_matrix_col(&self, idx: isize2, n: usize, j: usize) -> SpVec<R> {
        let ej = SpVec::unit(n, j);
        let z = self.as_chain(idx, &ej);        
        let w = self.d(idx, &z);
        self.vectorize(idx + self.d_deg(), &w)
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
        }
    }

    fn get(&self, i: isize2) -> &Self::Output { 
        &self.summands[i]
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
        self.get(i).rank()
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

    fn make_cpx(link: &Link, q: isize) -> KRTotComplex<R> {
        let data = Arc::new( KRCubeData::<R>::new(link) );
        KRTotComplex::new(data, q)
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
        let c = make_cpx(&l, q).as_generic().reduced(false);

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