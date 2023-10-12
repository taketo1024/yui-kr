use std::cell::OnceCell;
use std::collections::HashMap;
use std::rc::Rc;

use yui_core::{EucRing, EucRingOps};
use yui_homology::{Idx2Iter, Idx2, GenericChainComplex};
use yui_utils::bitseq::BitSeq;

use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

struct KRTotCube<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    data: Rc<KRCubeData<R>>,
    q_slice: isize,
    hor_hmls: HashMap<BitSeq, OnceCell<KRHorHomol<R>>>
} 

impl<R> KRTotCube<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Rc<KRCubeData<R>>, q_slice: isize) -> Self {
        let n = data.dim();
        let hor_homols = BitSeq::generate(n).into_iter().map(|v|
            (v, OnceCell::new())
        ).collect();

        Self {
            data,
            q_slice,
            hor_hmls: hor_homols
        }
    }

    fn hor_hml(&self, v_coords: BitSeq) -> &KRHorHomol<R> {
        &self.hor_hmls[&v_coords].get_or_init(|| { 
            KRHorHomol::new(self.data.clone(), v_coords, self.q_slice)
        })
    }

    pub fn as_complex(self) -> GenericChainComplex<R, Idx2Iter> {
        let n = self.data.dim() as isize;

        let start = Idx2(0, 0);
        let end   = Idx2(n, n);
        let range = start.iter_rect(end, (1, 1));
        
        GenericChainComplex::generate(range, Idx2(0, 1), |idx| {
            let (i, j) = idx.as_tuple();
            todo!()
        })
    }
}