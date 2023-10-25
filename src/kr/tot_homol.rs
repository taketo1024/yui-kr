use std::rc::Rc;
use yui_core::{EucRing, EucRingOps};
use yui_homology::Homology2;

use super::data::KRCubeData;
use super::tot_cube::KRTotCube;

pub(crate) struct KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    q_slice: isize,
    homology: Homology2<R>
} 

impl<R> KRTotHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, q_slice: isize) -> Self { 
        let cube = KRTotCube::new(data, q_slice);
        let complex = cube.as_complex();
        let reduced = complex.reduced(false);
        let homology = reduced.homology(false);

        Self { q_slice, homology }
    }

    pub fn rank(&self, i: usize, j: usize) -> usize { 
        let (i, j) = (i as isize, j as isize);
        self.homology[(i, j)].rank()
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