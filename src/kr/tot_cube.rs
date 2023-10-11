use std::cell::OnceCell;
use std::collections::HashMap;
use std::rc::Rc;

use yui_core::{Ring, RingOps};
use yui_utils::bitseq::BitSeq;

use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

struct KRTotCube<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    data: Rc<KRCubeData<R>>,
    q_slice: isize,
    hor_hmls: HashMap<BitSeq, OnceCell<KRHorHomol<R>>>
} 

impl<R> KRTotCube<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
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

    fn vert_gens(&self, h: usize, v_coords: BitSeq) {
        self.hor_hml(v_coords)[h]
    }
}