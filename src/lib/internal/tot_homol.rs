use std::ops::{Index, RangeInclusive};
use std::sync::Arc;
use delegate::delegate;

use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, GridTrait, GridIter, HomologySummand, isize3, Grid2, ChainComplex2, DisplayTable};

use super::tot_cpx::KRTotComplex;
use super::base::extend_ends_bounded;
use super::data::KRCubeData;

#[derive(Default)]
pub struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q: isize,
    inner: Grid2<HomologySummand<R>>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q: isize) -> Self { 
        let n = data.dim() as isize;
        Self::new_restr(data, q, (0..=n, 0..=n))
    }

    pub fn new_restr(data: Arc<KRCubeData<R>>, q: isize, range: (RangeInclusive<isize>, RangeInclusive<isize>)) -> Self { 
        let n = data.dim() as isize;
        let range = (range.0, extend_ends_bounded(range.1, 1, 0..=n));

        let complex = KRTotComplex::new_restr(
            data.clone(), 
            q, 
            range
        ).reduced();

        info!("H_tot (q: {})..", q);

        let inner = Grid2::generate(
            complex.support(), 
            |idx| Self::homology_at(&data, q, &complex, idx)
        );

        info!("H_tot (q: {})\n{}", q, inner.display_table("h", "v"));

        Self { q, inner }
    }

    fn homology_at(data: &KRCubeData<R>, q: isize, complex: &ChainComplex2<R>, idx: isize2) -> HomologySummand<R> { 
        let (i, j) = idx.into();
        let g = isize3(q, i, j);

        if data.is_triv_inner(g) { 
            HomologySummand::zero()
        } else { 
            complex.homology_at(idx, false)
        }
    }

    pub fn q_deg(&self) -> isize { 
        self.q
    }
}

impl<R> GridTrait<isize2> for KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize2>;
    type Output = HomologySummand<R>;

    delegate! { 
        to self.inner {
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, i: isize2) -> bool;
        }
    }

    fn get(&self, idx: isize2) -> &Self::Output { 
        &self.inner[idx]
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