use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;
use cartesian::cartesian;
use delegate::delegate;

use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, GridTrait, GridIter, HomologySummand, DisplayTable, isize3, ChainComplex, ChainComplexTrait, DisplaySeq, RModStr};

use crate::kr::tot_cpx::KRTotComplex;
use super::data::KRCubeData;

pub struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    data: Arc<KRCubeData<R>>,
    complex: KRTotComplex<R>,
    reduced: Vec<OnceCell<ChainComplex<R>>>, // vertical slices
    homology: HashMap<isize2, OnceCell<HomologySummand<R>>>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        info!("start tot-homol, q: {q_slice}.");

        let complex = KRTotComplex::new(data.clone(), q_slice, true); // skip triv
        info!("tot-complex, q: {q_slice}\n{}", complex.display_table());

        let n = data.dim() as isize;
        let reduced = (0..=n).map(|_| OnceCell::new() ).collect();
        let homology = cartesian!(0..=n, 0..=n).map( |(i, j)| 
            (isize2(i, j), OnceCell::new()) 
        ).collect();

        Self { q_slice, data, complex, reduced, homology }
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    fn reduced(&self, i: isize) -> &ChainComplex<R> { 
        let n = self.data.dim() as isize;
        self.reduced[i as usize].get_or_init(|| { 
            info!("reduce tot-complex, q: {}, i: {i}.", self.q_slice);

            let red = ChainComplex::generate(0..=n, 1, |j|
                self.complex.d_matrix(isize2(i, j))
            ).reduced(false);
            info!("reduced tot-complex, q: {}, i: {i}\n{}", self.q_slice, red.display_seq());

            red
        })
    }

    fn homology(&self, idx: isize2) -> &HomologySummand<R> { 
        self.homology[&idx].get_or_init(|| {
            let isize2(i, j) = idx;
            let g = isize3(self.q_slice, i, j);

            if self.data.is_triv_inner(g) { 
                HomologySummand::zero()
            } else { 
                info!("compute tot-homology, q: {}, {idx}.", self.q_slice);
                let h = self.reduced(i).homology_at(j, false);
                info!("tot-homology, q: {}, {idx} -> ", h.math_symbol());

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
        to self.complex {
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