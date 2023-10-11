use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::rc::Rc;
use itertools::Itertools;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{FreeChainComplex, ChainComplex, RModStr};
use yui_homology::utils::HomologyCalc;
use yui_matrix::sparse::{SpVec, SpMat};
use yui_utils::bitseq::BitSeq;
use crate::kr::hor_cube::KRHorCube;

use super::base::{VertGen, EdgeRing};
use super::data::KRCubeData;

type KRHorHomolSummand<R> = (Vec<(BitSeq, EdgeRing<R>)>, SpMat<R>);

pub(crate) struct KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    complex: FreeChainComplex<VertGen, R, RangeInclusive<isize>>,
    cache: HashMap<usize, OnceCell<KRHorHomolSummand<R>>>
} 

impl<R> KRHorHomol<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self { 
        let n = data.dim();
        let cube = KRHorCube::new(data, v_coords, q_slice);
        let complex = cube.as_complex();
        let cache = (0..=n).map(|i| 
            (i, OnceCell::new())
        ).collect();

        Self { complex, cache }
    }

    fn compute_summand(&self, j: usize) -> KRHorHomolSummand<R> { 
        let j = j as isize;
        let d1 = self.complex.d_matrix(j);
        let d2 = self.complex.d_matrix(j+1);

        // p: (r, n), q: (n, r)
        let (h, p, q) = HomologyCalc::calculate_with_trans(d1, d2);
        let r = h.rank();

        let gens = self.complex[j].generators();
        let h_gens = (0..r).map(|k| { 
            let v = q.col_vec(k); // 
            let e = v.iter().map(|(i, a)| {
                let x = &gens[i];
                todo!() // group by h-coords
            });
            
            todo!()
        }).collect();

        (h_gens, p)
    }

    fn summand(&self, j: usize) -> &KRHorHomolSummand<R> {
        &self.cache[&j].get_or_init(|| { 
            self.compute_summand(j)
        })
    }

    pub fn gens(&self, j: usize) -> &Vec<(BitSeq, EdgeRing<R>)> { 
        &self.summand(j).0
    }

    pub fn vectorize(&self, j: usize, p: EdgeRing<R>) -> SpVec<R> {
        todo!()
    }
}