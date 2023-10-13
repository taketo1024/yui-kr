use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;

use cartesian::{cartesian, TuplePrepend};
use itertools::Itertools;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{GenericHomology, Idx2Iter, HomologyComputable, Idx2, RModStr};
use yui_link::Link;
use yui_polynomial::LPoly;

use super::base::TripGrad;
use super::data::KRCubeData;
use super::tot_cube::KRTotCube;

type TotHomology<R> = GenericHomology<R, Idx2Iter>;
type QPoly<R> = LPoly<'q', R>;

struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Rc<KRCubeData<R>>,
    cache: UnsafeCell<HashMap<isize, TotHomology<R>>>
}

impl<R> KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(link: &Link) -> Self { 
        let data = Rc::new(KRCubeData::new(&link));
        let cache = UnsafeCell::new( HashMap::new() );
        Self { data, cache }
    }

    fn tot_hml(&self, q_slice: isize) -> &TotHomology<R> {
        let cache = unsafe { &mut *self.cache.get() };
        cache.entry(q_slice).or_insert_with(|| {
            let cube = KRTotCube::new(self.data.clone(), q_slice);
            let hml = cube.as_complex().homology();
            hml
        })
    }

    pub fn rank(&self, i: isize, j: isize, k: isize) -> usize { 
        if i > 0 { 
            return self.rank(-i, j, k + 2 * i)
        }
        
        let grad = TripGrad(i, j, k);
        if self.data.is_triv(grad) { 
            return 0
        }

        let Some(TripGrad(h, v, q)) = self.data.to_inner_grad(grad) else { 
            return 0
        };

        let hml = self.tot_hml(q);
        hml[Idx2(h, v)].rank()
    }

    pub fn rank_all(&self) -> HashMap<TripGrad, usize> { 
        let i_range = self.data.i_range().step_by(2);
        let j_range = self.data.j_range().step_by(2);
        let range = cartesian!(i_range.clone(), j_range.clone(), i_range.clone());

        let ranks = range.filter_map(|(i, j, k)| {
            let r = self.rank(i, j, k);
            if r > 0 { 
                Some((TripGrad(i, j, k), r))
            } else { 
                None
            }
        }).collect();

        ranks
    }

    pub fn qpoly_table(&self) -> HashMap<Idx2, QPoly<R>> { 
        let str = self.rank_all();        
        let polys = str.into_iter().into_group_map_by(|(idx, _)|
            Idx2(idx.1, idx.2) // (j, k)
        ).into_iter().map(|(jk, list)| { 
            let elems = list.into_iter().map(|(idx, r)| {
                let i = idx.0;
                let a = R::from(r as i32);
                (i, a) // a.q^i
            });
            let p = QPoly::from_deg_iter(elems);
            (jk, p)
        }).collect();

        polys
    }

    pub fn print_table(&self) { 
        let j_range = self.data.j_range().step_by(2);
        let k_range = self.data.i_range().rev().step_by(2);
        let polys = self.qpoly_table();   

        let table = yui_utils::table("k\\j", &k_range.collect(), &j_range.collect(), |k, j| { 
            if let Some(p) = polys.get(&Idx2(j, k)) { 
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
        let ranks = h.rank_all();

        // TODO
        println!("{ranks:?}");
    }
}