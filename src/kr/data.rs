use std::marker::PhantomData;

use yui_core::{Ring, RingOps};
use yui_link::Link;

pub(crate) struct KRCubeData<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    dim: usize,
    x_signs: Vec<i32>,
    _base_ring: PhantomData<R>
}

impl<R> KRCubeData<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(link: &Link) -> Self {
        let dim = link.crossing_num() as usize;
        let x_signs = link.crossing_signs();

        Self {
            dim,
            x_signs,
            _base_ring: PhantomData,
        }
    }

    pub fn dim(&self) -> usize { 
        self.dim
    }

    pub fn x_signs(&self) -> &Vec<i32> { 
        &self.x_signs
    }
}