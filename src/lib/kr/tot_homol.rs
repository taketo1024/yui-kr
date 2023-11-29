use std::cell::OnceCell;
use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, GridTrait, GridIter, HomologySummand, isize3, ChainComplex, ChainComplexTrait, DisplaySeq, RModStr, Grid2};

use crate::kr::tot_cpx::KRTotComplex;
use super::data::KRCubeData;

pub struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    data: Arc<KRCubeData<R>>,
    complex: KRTotComplex<R>,
    reduced: Vec<OnceCell<ChainComplex<R>>>, // vertical slices
    homology: Grid2<OnceCell<HomologySummand<R>>>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        info!("create H_tot (q: {q_slice}).");

        let n = data.dim() as isize;
        let complex = KRTotComplex::new(data.clone(), q_slice, true); // skip triv
        let reduced = (0..=n).map(|_| OnceCell::new() ).collect();    // lazy
        let homology = Grid2::generate(
            complex.support(), 
            |_| OnceCell::new()
        );

        Self { q_slice, data, complex, reduced, homology }
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    fn reduced(&self, i: isize) -> &ChainComplex<R> { 
        let n = self.data.dim() as isize;
        self.reduced[i as usize].get_or_init(|| { 
            let cpx = ChainComplex::generate(0..=n, 1, |j|
                self.complex.d_matrix(isize2(i, j))
            );
            info!("C_tot/h (q: {}, h: {i})\n{}", self.q_slice, cpx.display_seq());
            
            info!("reduce C_tot/h (q: {}, h: {i}).", self.q_slice);
            let red = cpx.reduced(false);
            info!("reduced C_tot/h (q: {}, h: {i})\n{}", self.q_slice, red.display_seq());

            red
        })
    }

    fn homology(&self, idx: isize2) -> &HomologySummand<R> { 
        self.homology[idx].get_or_init(|| {
            let isize2(i, j) = idx;
            let g = isize3(self.q_slice, i, j);

            if self.data.is_triv_inner(g) { 
                HomologySummand::zero()
            } else { 
                info!("compute H_tot (q: {}, h: {}, v: {}).", self.q_slice, idx.0, idx.1);
                let h = self.reduced(i).homology_at(j, false);
                info!("H_tot (q: {}, h: {}, v: {}) => {}", self.q_slice, idx.0, idx.1, h.math_symbol());

                h
            }
        })
    }
}

impl<R> GridTrait<isize2> for KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize2>;
    type Output = HomologySummand<R>;

    delegate! { 
        to self.homology {
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize2) -> bool;
        }
    }

    fn get(&self, idx: isize2) -> &Self::Output { 
        self.homology(idx)
    }
}

impl<R> Index<(isize, isize)> for KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = HomologySummand<R>;
    fn index(&self, index: (isize, isize)) -> &Self::Output { 
        self.get(index.into())
    }
}

#[cfg(test)]
mod tests { 
    use yui_homology::RModStr;
    use yui_link::Link;
    use yui::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn rank() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let data = Arc::new( KRCubeData::<R>::new(&l, 2) );
        let h = KRTotHomol::new(data, q);

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