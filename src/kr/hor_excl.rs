use std::collections::HashSet;
use std::hash::Hash;

use yui_core::{Ring, RingOps, IndexList};
use yui_homology::XModStr;
use yui_matrix::sparse::{Trans, SpMat};
use yui_polynomial::MonoGen;
use super::base::{EdgeRing, VertGen};

struct KRHorExclElem<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    h_index: usize,
    var_index: usize,
    deg: usize, 
    poly: EdgeRing<R>
}

pub(crate) struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    elements: Vec<KRHorExclElem<R>>,
    h_indices: HashSet<usize>,
    lin_exc_vars: HashSet<usize>,
    quad_exc_vars: HashSet<usize>
}

impl<R> KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn trans_for(&self, v: &XModStr<VertGen, R>) -> Trans<R> {
        let from = v.gens().iter().collect();
        let to = self.reduce_gens(&from);

        let fwd = make_matrix(&from, &to, |x| self.forward(x));
        let bwd = make_matrix(&to, &from, |x| self.backward(x));

        Trans::new(fwd, bwd)
    }

    fn reduce_gens<'a>(&self, gens: &IndexList<&'a VertGen>) -> IndexList<&'a VertGen> {
        gens.iter().filter_map(|&v| 
            if self.should_reduce(v) { 
                None
            } else {
                Some(v)
            }
        ).collect()
    }

    fn should_reduce(&self, v: &VertGen) -> bool { 
        let (h, x) = (&v.0, &v.2);
        
        self.h_indices.iter().any(|&i| h[i].is_zero()) || 
        x.deg().iter().any(|(i, &d_i)| 
            self.lin_exc_vars.contains(i) || 
            (self.quad_exc_vars.contains(i) && d_i >= 2)
        )
    }

    fn forward(&self, v: &VertGen) -> Vec<(VertGen, R)> {
        todo!()
    }

    fn backward(&self, w: &VertGen) -> Vec<(VertGen, R)> {
        todo!()
    }
}

fn rem<R>(f: EdgeRing<R>, g: &EdgeRing<R>, i: usize) -> EdgeRing<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    todo!()
}

fn make_matrix<'a, X, Y, R, F>(from: &IndexList<&X>, to: &IndexList<&Y>, f: F) -> SpMat<R>
where 
    X: Hash + Eq, Y: Hash + Eq, 
    R: Ring, for<'x> &'x R: RingOps<R>,
    F: Fn(&X) -> Vec<(Y, R)> 
{
    let (m, n) = (to.len(), from.len());
    SpMat::generate((m, n), |set|
        for (j, &x) in from.iter().enumerate() {
            let ys = f(x);
            for (y, a) in ys {
                let i = to.index_of(&&y).unwrap();
                set(i, j, a);
            }
        }
    )
}