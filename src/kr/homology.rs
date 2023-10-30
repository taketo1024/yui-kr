use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;

use cartesian::{cartesian, TuplePrepend};
use itertools::Itertools;
use yui_core::{EucRing, EucRingOps, isize2, isize3};
use yui_homology::{GridTrait, RModStr};
use yui_link::Link;
use yui_polynomial::LPoly;

use super::data::KRCubeData;
use super::tot_homol::KRTotHomol;

type QPoly<R> = LPoly<'q', R>;

pub struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Rc<KRCubeData<R>>,
    cache: UnsafeCell<HashMap<isize, KRTotHomol<R>>>
}

impl<R> KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(link: &Link) -> Self { 
        let data = Rc::new(KRCubeData::new(&link));
        let cache = UnsafeCell::new( HashMap::new() );
        Self { data, cache }
    }

    fn tot_hml(&self, q_slice: isize) -> &KRTotHomol<R> {
        let cache = unsafe { &mut *self.cache.get() };
        cache.entry(q_slice).or_insert_with(|| {
            KRTotHomol::new(self.data.clone(), q_slice)
        })
    }

    pub fn rank(&self, i: isize, j: isize, k: isize) -> usize { 
        if i > 0 { 
            return self.rank(-i, j, k + 2 * i)
        }
        
        let grad = isize3(i, j, k);
        if self.data.is_triv(grad) { 
            return 0
        }

        let Some(isize3(h, v, q)) = self.data.to_inner_grad(grad) else { 
            return 0
        };

        self.tot_hml(q).get(isize2(h, v)).rank()
    }

    pub fn rank_all(&self) -> HashMap<isize3, usize> { 
        let i_range = self.data.i_range().step_by(2);
        let j_range = self.data.j_range().step_by(2);
        let k_range = self.data.k_range().step_by(2);

        let range = cartesian!(i_range, j_range.clone(), k_range.clone());

        let ranks = range.filter_map(|(i, j, k)| {
            let r = self.rank(i, j, k);
            if r > 0 { 
                Some((isize3(i, j, k), r))
            } else { 
                None
            }
        }).collect();

        ranks
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
                "".to_string()
            }
        });

        println!("{table}");
    }
}

#[cfg(test)]
mod tests { 
    use yui_link::Link;
    use yui_ratio::Ratio;
    use crate::kr::base::EdgeRing;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

    #[test]
    fn rank() { 
        let l = Link::trefoil();
        let h = KRHomology::<R>::new(&l);

        // TODO
        h.print_table();
    }
}