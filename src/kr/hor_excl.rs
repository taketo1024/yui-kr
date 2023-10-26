use std::collections::{HashSet, HashMap};
use std::hash::Hash;

use yui_core::{Ring, RingOps};
use yui_matrix::sparse::{Trans, SpMat};
use yui_polynomial::MonoGen;
use super::base::{EdgeRing, VertGen};

pub struct Indexer<'a, V>
where V: Eq + Hash { 
    dict: HashMap<&'a V, usize>
}

impl<'a, V> Indexer<'a, V>
where V: Eq + Hash { 
    pub fn index_of(&self, v: &V) -> Option<usize> { 
        self.dict.get(v).cloned()
    }
}

impl<'a, V> From<&'a Vec<V>> for Indexer<'a, V>
where V: Eq + Hash {
    fn from(vec: &'a Vec<V>) -> Self {
        let dict = vec.iter().enumerate().map(|(i, v)| (v, i)).collect();
        Self { dict }
    }
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
    fn trans_for(&self, gens: &Vec<VertGen>) -> Trans<R> {
        let red_gens = self.reduce_gens(gens);
        let i_org = Indexer::from(gens);
        let i_red = Indexer::from(&red_gens);

        let n = gens.len();
        let m = red_gens.len();

        let fwd = SpMat::generate((m, n), |set| {
            for (j, v) in gens.iter().enumerate() {
                for (w, r) in self.forward(v).into_iter() {
                    let i = i_org.index_of(&w).unwrap();
                    set(i, j, r)
                }
            }
        });

        let bwd = SpMat::generate((n, m), |set| {
            for (j, w) in red_gens.iter().enumerate() {
                for (v, r) in self.backward(w).into_iter() {
                    let i = i_red.index_of(&&v).unwrap();
                    set(i, j, r)
                }
            }
        });

        Trans::new(fwd, bwd)
    }

    fn reduce_gens<'a>(&self, gens: &'a Vec<VertGen>) -> Vec<&'a VertGen> {
        gens.iter().filter(|v| 
            !self.should_reduce(v)
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

pub struct KRHorExclElem<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub h_index: usize,
    pub var_index: usize,
    pub deg: usize, 
    pub poly: EdgeRing<R>
}