use std::cell::OnceCell;
use std::ops::{Index, RangeInclusive};
use std::sync::Arc;
use delegate::delegate;

use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, GridTrait, GridIter, HomologySummand, isize3, ChainComplex, Grid2, Grid1};

use crate::kr::tot_cpx::KRTotComplex;
use super::base::extend_ends_bounded;
use super::data::KRCubeData;

pub struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    data: Arc<KRCubeData<R>>,
    complex: KRTotComplex<R>,
    reduced: Grid1<OnceCell<ChainComplex<R>>>, // vertical slices
    homology: Grid2<OnceCell<HomologySummand<R>>>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self { 
        let n = data.dim() as isize;
        Self::new_restr(data, q_slice, (0..=n, 0..=n))
    }

    pub fn new_restr(data: Arc<KRCubeData<R>>, q_slice: isize, range: (RangeInclusive<isize>, RangeInclusive<isize>)) -> Self { 
        let n = data.dim() as isize;
        let range = (range.0, extend_ends_bounded(range.1, 1, 0..=n));
        let complex = KRTotComplex::new_restr(
            data.clone(), 
            q_slice, 
            range.clone()
        );
        let reduced = Grid1::generate(
            range.0.clone(), 
            |_| OnceCell::new()
        );
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
        self.reduced[i].get_or_init(|| { 
            self.complex.h_slice(i).reduced(false)
        })
    }

    fn homology(&self, idx: isize2) -> &HomologySummand<R> { 
        self.homology[idx].get_or_init(|| {
            let (i, j) = idx.into();
            let g = isize3(self.q_slice, i, j);

            if self.data.is_triv_inner(g) { 
                HomologySummand::zero()
            } else { 
                self.reduced(i).homology_at(j, false)
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
        let data = Arc::new( KRCubeData::<R>::new(&l) );
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