use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

use cartesian::{cartesian, TuplePrepend};
use itertools::Itertools;
use yui_core::{EucRing, EucRingOps, isize2, isize3};
use yui_homology::{GridTrait, RModStr, GridIter, HomologySummand};
use yui_link::Link;
use yui_polynomial::LPoly;

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
        let data = Arc::new(KRCubeData::new(&link));
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
        let ranks = self.support().filter_map(|isize3(i, j, k)| {
            let r = self[(i, j, k)].rank();
            if r > 0 { 
                Some((isize3(i, j, k), r))
            } else { 
                None
            }
        }).collect();

        ranks
    }

    pub fn tot_rank(&self) -> usize { 
        self.rank_all().iter().map(|(_, r)| r).sum()
    }

    pub fn qpoly_table(&self) -> HashMap<isize2, QPoly<R>> { 
        let str = self.rank_all();        
        let polys = str.into_iter().into_group_map_by(|(idx, _)|
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
        }).collect();

        polys
    }

    pub fn print_table(&self) { 
        let j_range = self.data.j_range().step_by(2);
        let k_range = self.data.k_range().rev().step_by(2);
        let polys = self.qpoly_table();   

        let table = yui_utils::table("k\\j", k_range, j_range, |&k, &j| { 
            if let Some(p) = polys.get(&isize2(j, k)) { 
                p.to_string()
            } else { 
                ".".to_string()
            }
        });

        println!("{table}");
    }
}

impl<R> GridTrait<isize3> for KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Itr = GridIter<isize3>;
    type E = HomologySummand<R>;

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

    fn get(&self, idx: isize3) -> &Self::E {
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
    use yui_link::Link;
    use yui_ratio::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn rank() { 
        let l = Link::trefoil();
        let h = KRHomology::<R>::new(&l);

        assert_eq!(h.tot_rank(), 3);
        assert_eq!(h[( 0,-4, 2)].rank(), 1);
        assert_eq!(h[(-2,-2, 2)].rank(), 1);
        assert_eq!(h[( 2,-2,-2)].rank(), 1);
    }
}