use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::rc::Rc;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{FreeChainComplex, ChainComplex, RModStr};
use yui_homology::utils::HomologyCalc;
use yui_lin_comb::LinComb;
use yui_matrix::sparse::SpMat;
use yui_utils::bitseq::BitSeq;
use crate::kr::hor_cube::KRHorCube;

use super::base::VertGen;
use super::data::KRCubeData;

type KRHorHomolSummand<R> = (Vec<LinComb<VertGen, R>>, SpMat<R>);

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

    fn compute_summand(&self, i: usize) -> KRHorHomolSummand<R> { 
        let j = i as isize;
        let d1 = self.complex.d_matrix(j);
        let d2 = self.complex.d_matrix(j+1);

        // p: (r, n), q: (n, r)
        let (h, p, q) = HomologyCalc::calculate_with_trans(d1, d2);
        let r = h.rank();

        let gens = self.complex[j].generators();
        let h_gens = (0..r).map(|k| { 
            let v = q.col_vec(k); 
            let terms = v.iter().map(|(i, a)| {
                let x = &gens[i];
                (x.clone(), a.clone())
            });
            LinComb::from_iter(terms)
        }).collect();

        (h_gens, p)
    }

    fn summand(&self, i: usize) -> &KRHorHomolSummand<R> {
        &self.cache[&i].get_or_init(|| { 
            self.compute_summand(i)
        })
    }

    pub fn rank(&self, i: usize) -> usize { 
        self.summand(i).0.len()
    }

    pub fn gens(&self, i: usize) -> &Vec<LinComb<VertGen, R>> { 
        &self.summand(i).0
    }
}