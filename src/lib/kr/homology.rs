use std::cell::OnceCell;
use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use delegate::delegate;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GridTrait, RModStr, GridIter, HomologySummand, Grid1};
use yui_link::Link;

use super::data::KRCubeData;
use super::result::KRHomologyStr;
use super::tot_homol::KRTotHomol;

pub struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    slices: Grid1<OnceCell<KRTotHomol<R>>>,
    zero: HomologySummand<R>
}

impl<R> KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(link: &Link) -> Self { 
        let excl_level = 2;
        let data = KRCubeData::new(link, excl_level);
        Self::from_data(data)
    }

    pub fn from_data(data: KRCubeData<R>) -> Self { 
        let data = Arc::new(data);
        let slices = Grid1::generate(data.q_range(), |_| OnceCell::new());
        let zero = HomologySummand::zero();
        Self { data, slices, zero }
    }

    delegate! { 
        to self.data { 
            pub fn i_range(&self) -> RangeInclusive<isize>;
            pub fn j_range(&self) -> RangeInclusive<isize>;
            pub fn k_range(&self) -> RangeInclusive<isize>;
            pub fn q_range(&self) -> RangeInclusive<isize>;
        }
    }

    fn slice(&self, q_slice: isize) -> &KRTotHomol<R> {
        self.slices[q_slice].get_or_init(||
            KRTotHomol::new(self.data.clone(), q_slice)
        )
    }

    pub fn structure(&self) -> KRHomologyStr { 
        self.support().filter_map(|idx| {
            let h = self.get(idx);
            let r = h.rank();

            if r > 0 { 
                Some(((idx.0, idx.1, idx.2), r))
            } else { 
                None
            }
        }).collect()
    }
}

impl<R> GridTrait<isize3> for KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize3>;
    type Output = HomologySummand<R>;

    delegate! { 
        to self.data { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, idx: isize3) -> bool;
        }
    }

    fn get(&self, idx: isize3) -> &Self::Output {
        if !self.is_supported(idx) { 
            return &self.zero
        }

        let Some(isize3(q, h, v)) = self.data.to_inner_grad(idx) else { 
            return &self.zero
        };

        self.slice(q).get(isize2(h, v))
    }
}

impl<R> Index<(isize, isize, isize)> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = HomologySummand<R>;
    fn index(&self, index: (isize, isize, isize)) -> &Self::Output {
        self.get(index.into())
    }
}

#[cfg(test)]
mod tests { 
    use yui_link::Braid;
    use yui::Ratio;
    use yui::util::macros::hashmap;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn b3_1() { 
        let b = Braid::from([1,1,1]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 3);
        assert_eq!(s.inner(), &hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        });
    }
    
    #[test]
    fn b4_1() { 
        let b = Braid::from([1,-2,1,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s.inner(), &hashmap!{ 
            (0,-2,2) => 1,
            (-2,0,2) => 1,
            (0,2,-2) => 1,
            (0,0,0)  => 1,
            (2,0,-2) => 1
        });
    }

    #[test]
    fn b5_1() { 
        let b = Braid::from([1,1,1,1,1]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s.inner(), &hashmap!{             
            (0,4,0)  => 1,
            (-2,6,0) => 1,
            (-4,4,4) => 1,
            (4,4,-4) => 1,
            (2,6,-4) => 1
        });
    }

    #[test]
    fn b5_2() { 
        let b = Braid::from([1,1,1,2,-1,2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 7);
        assert_eq!(s.inner(), &hashmap!{ 
            (2,4,-4) => 1,
            (2,2,-2) => 1,
            (0,4,-2) => 1,
            (-2,4,0) => 1,
            (0,2,0)  => 1,
            (0,6,-4) => 1,
            (-2,2,2) => 1
        });
    }

    #[test]
    fn b6_1() { 
        let b = Braid::from([1,1,2,-1,-3,2,-3]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 9);
        assert_eq!(s.inner(), &hashmap!{ 
            (0,-2,2) => 1,
            (0,0,0)  => 2,
            (2,0,-2) => 1,
            (2,2,-4) => 1,
            (0,4,-4) => 1,
            (0,2,-2) => 1,
            (-2,2,0) => 1,
            (-2,0,2) => 1    
        })
    }

    #[test]
    fn b6_2() { 
        let b = Braid::from([1,1,1,-2,1,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 11);
        assert_eq!(s.inner(), &hashmap!{ 
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
        });
    }

    #[test]
    fn b6_3() { 
        let b = Braid::from([1,1,-2,1,-2,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);
        let s = h.structure();

        assert_eq!(s.total_rank(), 13);
        assert_eq!(s.inner(), &hashmap!{ 
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
        });
    }
}