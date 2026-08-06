use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use delegate::delegate;
use yui_core::abst::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GenericSummand, Grid1};
use yui_link::{Link, Braid};

use crate::KRHomologyStr;
use crate::internal::data::KRCubeData;
use crate::internal::tot_homol::KRTotHomol;

pub struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    q_slices: Grid1<KRTotHomol<R>>,
    zero: GenericSummand<isize2, R>,
}

impl<R> KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn from_braid(b: &Braid) -> Self {
        let l = b.closure();
        Self::from_link(&l)
    }

    /// note: H(D) is defined for any diagram, but is an invariant of the knot
    /// only when D is a braid closure [NS2024, Remark 2.2]. A minimal-crossing
    /// diagram may give a different answer -- 5_2 is such a case [NS2024, §5].
    pub fn from_link(link: &Link) -> Self {
        assert_eq!(link.n_comps(), 1, "only knots are supported.");
        let data = Arc::new(KRCubeData::new(link));
        Self::new(data)
    }

    pub fn new(data: Arc<KRCubeData<R>>) -> Self {
        let q_range = data.q_range();
        let q_slices = q_range.map(|q| {
            let range = data.hv_range(q);
            (q, KRTotHomol::new_restr(data.clone(), q, range))
        }).collect();
        let zero = GenericSummand::zero();
        Self { data, q_slices, zero }
    }

    delegate! { 
        to self.data { 
            pub fn i_range(&self) -> RangeInclusive<isize>;
            pub fn j_range(&self) -> RangeInclusive<isize>;
            pub fn k_range(&self) -> RangeInclusive<isize>;
            pub fn q_range(&self) -> RangeInclusive<isize>;
        }
    }

    pub fn structure(&self) -> KRHomologyStr {
        self.support().filter_map(|&idx| {
            let r = self[idx].rank();

            if r > 0 {
                Some(((idx.0, idx.1, idx.2), r))
            } else {
                None
            }
        }).collect()
    }

    pub fn to_table_string(&self) -> String { 
        self.structure().qpoly_table()
    }
}

impl<R> KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    delegate! {
        to self.data {
            pub fn support(&self) -> std::collections::hash_set::Iter<'_, isize3>;
            pub fn is_supported(&self, idx: isize3) -> bool;
        }
    }
}

impl<R> Index<isize3> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = GenericSummand<isize2, R>;
    fn index(&self, idx: isize3) -> &Self::Output {
        match self.data.to_inner_grad(idx) {
            Some(isize3(q, h, v)) => &self.q_slices[q][isize2(h, v)],
            None => &self.zero
        }
    }
}

impl<R> Index<(isize, isize, isize)> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = GenericSummand<isize2, R>;
    fn index(&self, index: (isize, isize, isize)) -> &Self::Output {
        &self[isize3::from(index)]
    }
}

#[cfg(test)]
mod tests { 
    use maplit::hashmap;
    use yui_link::Braid;
    use yui_core::num::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn trivial() { 
        let b = Braid::from([1]);
        let h = KRHomology::<R>::from_braid(&b);

        assert_eq!(h[(0,0,0)].rank(), 1);
        assert_eq!(h[(1,0,0)].rank(), 0);

        let s = h.structure();

        assert_eq!(s.total_rank(), 1);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,0,0) => 1
        }));
    }

    #[test]
    fn b3_1() { 
        let b = Braid::from([1,1,1]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 3);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        }));
    }
    
    #[test]
    fn b4_1() { 
        let b = Braid::from([1,-2,1,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,-2,2) => 1,
            (-2,0,2) => 1,
            (0,2,-2) => 1,
            (0,0,0)  => 1,
            (2,0,-2) => 1
        }));
    }

    #[test]
    fn b5_1() { 
        let b = Braid::from([1,1,1,1,1]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s, KRHomologyStr::from(hashmap!{             
            (0,4,0)  => 1,
            (-2,6,0) => 1,
            (-4,4,4) => 1,
            (4,4,-4) => 1,
            (2,6,-4) => 1
        }));
    }

    #[test]
    fn b5_2() { 
        let b = Braid::from([1,1,1,2,-1,2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 7);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (2,4,-4) => 1,
            (2,2,-2) => 1,
            (0,4,-2) => 1,
            (-2,4,0) => 1,
            (0,2,0)  => 1,
            (0,6,-4) => 1,
            (-2,2,2) => 1
        }));
    }

    #[test]
    fn b6_1() { 
        let b = Braid::from([1,1,2,-1,-3,2,-3]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 9);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,-2,2) => 1,
            (0,0,0)  => 2,
            (2,0,-2) => 1,
            (2,2,-4) => 1,
            (0,4,-4) => 1,
            (0,2,-2) => 1,
            (-2,2,0) => 1,
            (-2,0,2) => 1    
        }));
    }

    #[test]
    fn b6_2() { 
        let b = Braid::from([1,1,1,-2,1,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 11);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (2,4,-4) => 1,
            (-2,4,0) => 1,
            (0,2,0)  => 2,
            (2,2,-2) => 1,
            (2,0,0)  => 1,
            (-2,2,2) => 1,
            (-2,0,4) => 1,
            (4,2,-4) => 1,
            (-4,2,4) => 1
        }));
    }

    #[test]
    fn b6_3() { 
        let b = Braid::from([1,1,-2,1,-2,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 13);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (4,0,-4)  => 1,
            (2,-2,0)  => 1,
            (0,0,0)   => 3,
            (2,2,-4)  => 1,
            (-2,2,0)  => 1,
            (0,2,-2)  => 1,
            (0,-2,2)  => 1,
            (-2,0,2)  => 1,
            (-4,0,4)  => 1,
            (2,0,-2)  => 1,
            (-2,-2,4) => 1
        }));
    }
}