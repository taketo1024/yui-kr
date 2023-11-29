use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use delegate::delegate;
use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GridTrait, RModStr, GridIter, HomologySummand, Grid1};
use yui_link::Link;

use super::data::KRCubeData;
use super::tot_homol::KRTotHomol;

pub type KRHomologyStr = HashMap<(isize, isize, isize), usize>;

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
        let data = Arc::new(KRCubeData::new(link, excl_level));
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

    fn compute(&self, idx: isize3) -> &HomologySummand<R> {
        if !self.is_supported(idx) { 
            return &self.zero
        }

        let Some(isize3(q, h, v)) = self.data.to_inner_grad(idx) else { 
            return &self.zero
        };

        self.slice(q).get(isize2(h, v))
    }

    fn compute_all(&self) {
        info!("compute KRHomology.");
        
        info!("i-range: {:?}", self.i_range());
        info!("j-range: {:?}", self.j_range());
        info!("k-range: {:?}", self.k_range());
        info!("q-range: {:?}", self.q_range());
        
        info!("targets:");
        self.support().enumerate().for_each(|(i, idx)| 
            info!("  {}. {idx}", i + 1)
        );

        info!("- - - - - - - - - - - - - - - -");

        let total = self.support().count();
        self.support().enumerate().for_each(|(i, idx)| {
            info!("({}/{}) H[{}] ..", i + 1, total, idx);

            let h = self.compute(idx);
            
            info!("H[{}] => {}", idx, h.math_symbol());
            info!("- - - - - - - - - - - - - - - -");
        });
    }

    pub fn structure(&self) -> KRHomologyStr { 
        self.compute_all();
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

    delegate! { 
        to self.data { 
            fn support(&self) -> Self::Itr;
            fn is_supported(&self, idx: isize3) -> bool;
        }
    }

    fn get(&self, idx: isize3) -> &Self::Output {
        self.compute(idx)
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