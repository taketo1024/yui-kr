use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use cartesian::{cartesian, TuplePrepend};
use itertools::Itertools;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GridTrait, RModStr, GridIter, HomologySummand, DisplayTable};
use yui_link::Link;
use yui::poly::LPoly;

use super::data::KRCubeData;
use super::tot_homol::KRTotHomol;

type QPoly<R> = LPoly<'q', R>;

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

    fn rank_all(&self) -> HashMap<isize3, usize> { 
        self.support().filter_map(|isize3(i, j, k)| {
            let r = self[(i, j, k)].rank();
            if r > 0 { 
                Some((isize3(i, j, k), r))
            } else { 
                None
            }
        }).collect()
    }

    pub fn tot_rank(&self) -> usize { 
        self.rank_all().values().sum()
    }

    pub fn qpoly_table(&self) -> HashMap<isize2, QPoly<R>> { 
        let str = self.rank_all();        
        let elements = str.into_iter().into_group_map_by(|(idx, _)|
            isize2(idx.1, idx.2) // (j, k)
        ).into_iter().map(|(jk, list)| { 
            let q = QPoly::mono;
            let elems = list.into_iter().map(|(idx, r)| {
                let i = idx.0;
                let a = R::from(r as i32);
                (q(i), a) // a.q^i
            });
            let p = QPoly::from_iter(elems);
            (jk, p)
        });
        elements.collect()
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

impl<R> DisplayTable<isize2> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    fn display_table(&self) -> String {
        let polys = self.qpoly_table();   
        let j_range = range(polys.keys().map(|idx| idx.0)).step_by(2);
        let k_range = range(polys.keys().map(|idx| idx.1)).rev().step_by(2);

        yui::util::format::table("k\\j", k_range, j_range, |&k, &j| { 
            if let Some(p) = polys.get(&isize2(j, k)) { 
                p.to_string()
            } else { 
                ".".to_string()
            }
        })
    }
}

fn range<Itr>(itr: Itr) -> RangeInclusive<isize>
where Itr: Iterator<Item = isize> {
    if let Some((l, r)) = itr.fold(None, |res, i| { 
        if let Some((mut l, mut r)) = res { 
            if i < l { l = i }
            if r < i { r = i }
            Some((l, r))
        } else { 
            Some((i, i))
        }
    }) { 
        l..=r
    } else { 
        0..=0
    }
}

#[cfg(test)]
mod tests { 
    use yui_link::Braid;
    use yui::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn b3_1() { 
        let b = Braid::from([1,1,1]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 3);

        assert_eq!(h[(0,4,-2)].rank(), 1);
        assert_eq!(h[(-2,2,2)].rank(), 1);
        assert_eq!(h[(2,2,-2)].rank(), 1);
    }
    
    #[test]
    fn b4_1() { 
        let b = Braid::from([1,-2,1,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 5);

        assert_eq!(h[(0,-2,2)].rank(), 1);
        assert_eq!(h[(-2,0,2)].rank(), 1);
        assert_eq!(h[(0,2,-2)].rank(), 1);
        assert_eq!(h[(0,0,0)].rank(), 1);
        assert_eq!(h[(2,0,-2)].rank(), 1);
    }

    #[test]
    fn b5_1() { 
        let b = Braid::from([1,1,1,1,1]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 5);

        assert_eq!(h[(0,4,0)].rank(), 1);
        assert_eq!(h[(-2,6,0)].rank(), 1);
        assert_eq!(h[(-4,4,4)].rank(), 1);
        assert_eq!(h[(4,4,-4)].rank(), 1);
        assert_eq!(h[(2,6,-4)].rank(), 1);
    }

    #[test]
    fn b5_2() { 
        let b = Braid::from([1,1,1,2,-1,2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 7);

        assert_eq!(h[(2,4,-4)].rank(), 1);
        assert_eq!(h[(2,2,-2)].rank(), 1);
        assert_eq!(h[(0,4,-2)].rank(), 1);
        assert_eq!(h[(-2,4,0)].rank(), 1);
        assert_eq!(h[(0,2,0)].rank(), 1);
        assert_eq!(h[(0,6,-4)].rank(), 1);
        assert_eq!(h[(-2,2,2)].rank(), 1);
    }

    #[test]
    #[ignore]
    fn b6_1() { 
        let b = Braid::from([1,1,2,-1,-3,2,-3]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        h.print_table();

        assert_eq!(h.tot_rank(), 9);

        assert_eq!(h[(0,-2,2)].rank(), 1);
        assert_eq!(h[(0,0,0)].rank(), 2);
        assert_eq!(h[(2,0,-2)].rank(), 1);
        assert_eq!(h[(2,2,-4)].rank(), 1);
        assert_eq!(h[(0,4,-4)].rank(), 1);
        assert_eq!(h[(0,2,-2)].rank(), 1);
        assert_eq!(h[(-2,2,0)].rank(), 1);
        assert_eq!(h[(-2,0,2)].rank(), 1);
    }

    #[test]
    #[ignore]
    fn b6_2() { 
        let b = Braid::from([1,1,1,-2,1,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 11);

        assert_eq!(h[(0,4,-2)].rank(), 1);
        assert_eq!(h[(2,4,-4)].rank(), 1);
        assert_eq!(h[(-2,4,0)].rank(), 1);
        assert_eq!(h[(0,2,0)].rank(), 2);
        assert_eq!(h[(2,2,-2)].rank(), 1);
        assert_eq!(h[(2,0,0)].rank(), 1);
        assert_eq!(h[(-2,2,2)].rank(), 1);
        assert_eq!(h[(-2,0,4)].rank(), 1);
        assert_eq!(h[(4,2,-4)].rank(), 1);
        assert_eq!(h[(-4,2,4)].rank(), 1);
    }

    #[test]
    #[ignore]
    fn b6_3() { 
        let b = Braid::from([1,1,-2,1,-2,-2]);
        let l = b.closure();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 13);

        assert_eq!(h[(4,0,-4)].rank(), 1);
        assert_eq!(h[(2,-2,0)].rank(), 1);
        assert_eq!(h[(0,0,0)].rank(), 3);
        assert_eq!(h[(2,2,-4)].rank(), 1);
        assert_eq!(h[(-2,2,0)].rank(), 1);
        assert_eq!(h[(0,2,-2)].rank(), 1);
        assert_eq!(h[(0,-2,2)].rank(), 1);
        assert_eq!(h[(-2,0,2)].rank(), 1);
        assert_eq!(h[(-4,0,4)].rank(), 1);
        assert_eq!(h[(2,0,-2)].rank(), 1);
        assert_eq!(h[(-2,-2,4)].rank(), 1);
    }
}