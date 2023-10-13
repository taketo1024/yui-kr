use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::rc::Rc;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{RModStr, GenericRModStr, Idx2, HomologyComputable};
use super::data::KRCubeData;
use super::tot_cube::{KRTotCube, KRTotComplex};

type KRTotHomolSummand<R> = GenericRModStr<R>;

pub(crate) struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    complex: KRTotComplex<R>,
    cache: UnsafeCell<HashMap<Idx2, KRTotHomolSummand<R>>>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, q_slice: isize) -> Self { 
        let cube = KRTotCube::new(data, q_slice);
        let complex = cube.as_complex();
        let cache = UnsafeCell::new( HashMap::new() );

        Self { q_slice, complex, cache }
    }

    fn summand(&self, i: usize, j: usize) -> &KRTotHomolSummand<R> {
        let idx = Idx2(i as isize, j as isize);
        let cache = unsafe { &mut *self.cache.get() };
        cache.entry(idx).or_insert_with(|| {
            self.complex.homology_at(idx)
        })
    }

    pub fn rank(&self, i: usize, j: usize) -> usize { 
        self.summand(i, j).rank()
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
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let data = Rc::new( KRCubeData::<R>::new(&l) );
        let hml = KRTotHomol::new(data, q);

        assert_eq!(hml.rank(0, 0), 0);
        assert_eq!(hml.rank(0, 1), 0);
        assert_eq!(hml.rank(0, 2), 0);
        assert_eq!(hml.rank(0, 3), 0);
        assert_eq!(hml.rank(1, 0), 0);
        assert_eq!(hml.rank(1, 1), 0);
        assert_eq!(hml.rank(1, 2), 0);
        assert_eq!(hml.rank(1, 3), 0);
        assert_eq!(hml.rank(2, 0), 0);
        assert_eq!(hml.rank(2, 1), 0);
        assert_eq!(hml.rank(2, 2), 0);
        assert_eq!(hml.rank(2, 3), 1);
        assert_eq!(hml.rank(3, 0), 0);
        assert_eq!(hml.rank(3, 1), 1);
        assert_eq!(hml.rank(3, 2), 0);
        assert_eq!(hml.rank(3, 3), 0);
     }
}