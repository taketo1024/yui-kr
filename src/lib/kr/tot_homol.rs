use std::ops::Index;
use std::sync::Arc;
use delegate::delegate;

use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, Homology2, GridTrait, GridIter, HomologySummand, DisplayTable, isize3};

use crate::kr::tot_cpx::KRTotComplex;
use super::data::KRCubeData;

pub struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    homology: Homology2<R>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        info!("start tot-homol, q: {q_slice}.");

        let complex = KRTotComplex::new(data.clone(), q_slice, true); // skip triv
        info!("tot-complex, q: {q_slice}\n{}", complex.display_table());

        info!("reduce tot-complex, q: {q_slice}.");
        let reduced = complex.reduced();
        info!("reduced tot-complex, q: {q_slice}\n{}", reduced.display_table());
        
        info!("compute tot-homology, q: {q_slice}");
        let homology = Homology2::generate(
            reduced.support(),
            move |idx| { 
                if data.is_triv_inner(isize3(q_slice, idx.0, idx.1)) { 
                    HomologySummand::zero()
                } else { 
                    reduced.homology_at(idx, false)
                }
            }
        );
        info!("tot-homology, q: {q_slice} \n{}", homology.display_table());

        Self { q_slice, homology }
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
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
            fn get(&self, i: isize2) -> &Self::Output;
        }
    }
}

impl<R> Index<(isize, isize)> for KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = HomologySummand<R>;

    delegate! { 
        to self.homology { 
            fn index(&self, index: (isize, isize)) -> &Self::Output;
        }
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