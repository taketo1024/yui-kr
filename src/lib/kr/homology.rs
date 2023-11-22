use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

use cartesian::{cartesian, TuplePrepend};
use itertools::Itertools;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GridTrait, RModStr, GridIter, HomologySummand};
use yui_link::Link;

use super::data::KRCubeData;
use super::tot_homol::KRTotHomol;

pub type KRHomologyStr = HashMap<(isize, isize, isize), usize>;

pub struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    cache: UnsafeCell<HashMap<isize, KRTotHomol<R>>>,
    zero: HomologySummand<R>
}

impl<R> KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(link: &Link) -> Self { 
        let excl_level = 2;
        let data = Arc::new(KRCubeData::new(link, excl_level));
        let cache = UnsafeCell::new( HashMap::new() );
        let zero = HomologySummand::zero();
        Self { data, cache, zero }
    }

    fn tot_hml(&self, q_slice: isize) -> &KRTotHomol<R> {
        let cache = unsafe { &mut *self.cache.get() };
        cache.entry(q_slice).or_insert_with(|| {
            KRTotHomol::new(self.data.clone(), q_slice)
        })
    }

    pub fn structure(&self) -> KRHomologyStr { 
        self.support().filter_map(|isize3(i, j, k)| {
            let r = self[(i, j, k)].rank();
            if r > 0 { 
                Some(((i, j, k), r))
            } else { 
                None
            }
        }).collect()
    }

    pub fn tot_rank(&self) -> usize { 
        self.structure().values().sum()
    }

    pub fn display_table(&self) -> String {
        use crate::util::make_qpoly_table;
        let str = self.structure();
        make_qpoly_table(&str)
    }
}

impl<R> GridTrait<isize3> for KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize3>;
    type Output = HomologySummand<R>;

    fn support(&self) -> Self::Itr {
        let i_range = self.data.i_range().step_by(2);
        let j_range = self.data.j_range().step_by(2);
        let k_range = self.data.k_range().step_by(2);
        let range = cartesian!(i_range, j_range.clone(), k_range.clone());
        
        range.map(|idx| 
            idx.into()
        ).filter(|&idx| 
            self.is_supported(idx)
        ).collect_vec().into_iter()
    }

    fn is_supported(&self, idx: isize3) -> bool {
        !self.data.is_triv(idx)
    }

    fn get(&self, idx: isize3) -> &Self::Output {
        let isize3(i, j, k) = idx;
        if i > 0 { 
            return self.get(isize3(-i, j, k + 2 * i))
        }
        
        if !self.is_supported(idx) { 
            return &self.zero
        }

        let isize3(h, v, q) = self.data.to_inner_grad(idx).unwrap();

        self.tot_hml(q).get(isize2(h, v))
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

        assert_eq!(h.tot_rank(), 3);
        assert_eq!(h.structure(), hashmap!{ 
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

        assert_eq!(h.tot_rank(), 5);
        assert_eq!(h.structure(), hashmap!{ 
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

        assert_eq!(h.tot_rank(), 5);
        assert_eq!(h.structure(), hashmap!{             
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

        assert_eq!(h.tot_rank(), 7);
        assert_eq!(h.structure(), hashmap!{ 
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

        assert_eq!(h.tot_rank(), 9);
        assert_eq!(h.structure(), hashmap!{ 
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

        assert_eq!(h.tot_rank(), 11);
        assert_eq!(h.structure(), hashmap!{ 
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

        assert_eq!(h.tot_rank(), 13);
        assert_eq!(h.structure(), hashmap!{ 
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