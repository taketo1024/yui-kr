use std::collections::{HashSet, HashMap};
use std::hash::Hash;

use itertools::Itertools;
use yui_core::{Ring, RingOps, IndexList};
use yui_homology::XModStr;
use yui_link::Edge;
use yui_matrix::sparse::{Trans, SpMat};
use yui_polynomial::MonoGen;
use super::base::{EdgeRing, VertGen};
use super::hor_cube::KRHorCube;

struct KRHorExclElem<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    dir: usize, // direction of edge = index of crossing
    var: usize, // index of excluded var
    deg: usize, // either 1 or 2
    poly: EdgeRing<R> // divisor polynomial
}

pub(crate) struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    level: usize,
    edge_polys: HashMap<usize, EdgeRing<R>>,
    elements: Vec<KRHorExclElem<R>>,
    exc_dirs: HashSet<usize>,
    fst_exc_vars: HashSet<usize>,
    snd_exc_vars: HashSet<usize>,
    remain_vars: HashSet<usize>,
}

impl<R> KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(n: usize, edge_polys: HashMap<usize, EdgeRing<R>>, level: usize) -> Self { 
        assert!(level <= 2);
        Self { 
            level,
            edge_polys,
            elements: vec![], 
            exc_dirs: HashSet::new(), 
            fst_exc_vars: HashSet::new(), 
            snd_exc_vars: HashSet::new(),
            remain_vars: (0..n).collect(),
        }
    }

    pub fn from(cube: KRHorCube<R>, level: usize) -> Self { 
        assert!(level <= 2);

        let n = cube.data().dim();
        let edge_polys = (0..n).map(|i| 
            (i, cube.edge_poly_dir(i))
        ).collect();

        let mut res = Self::new(n, edge_polys, level);

        for d in 1..=level { 
            res.process_excl(d);
        }

        res
    }

    fn process_excl(&mut self, deg: usize) {
        assert!(deg == 1 || deg == 2);

        let inds = self.edge_polys.keys().sorted().cloned().collect_vec();
        for i in inds { 
            if let Some(k) = self.find_excl_var(deg, i) { 
                self.update_excl(deg, i, k);
            }
        }
    }

    fn find_excl_var(&self, deg: usize, i: usize) -> Option<usize> { 
        if self.remain_vars.is_empty() { 
            return None;
        }

        let p = &self.edge_polys[&i];

        // search for the term (x_k)^d in p.
        for (x, _) in p.iter() {
            let mdeg = x.deg();
            if mdeg.total() != deg { continue; }
            for &k in self.remain_vars.iter() {
                if mdeg.deg(k) == deg { 
                    return Some(k)
                }
            }
        }

        None
    }

    fn update_excl(&mut self, deg: usize, i: usize, k: usize) {
        debug_assert!(deg == 1 || deg == 2);

        // update polys
        let p = self.edge_polys.remove(&k).unwrap();
        self.edge_polys = self.edge_polys.iter().map(|(&j, f)|
            (j, rem(f, &p, i))
        ).collect();

        // update indep vars
        if deg == 1 { 
            // vars except x_k remain indep.
            self.remain_vars.remove(&k);
        } else { 
            // vars appearing in p are no longer indep. 
            for x in p.iter().map(|(x, _)| x) {
                for l in x.deg().iter().map(|(l, _)| l) {
                    self.remain_vars.remove(&l);
                }
            }
        }

        // insert results
        let e = KRHorExclElem { dir: i, var: k, deg, poly: p };
        self.elements.push(e);
        self.exc_dirs.insert(i);
        
        if deg == 1 { 
            self.fst_exc_vars.insert(k);
        } else { 
            self.snd_exc_vars.insert(k);
        }
    }

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
        
        self.exc_dirs.iter().any(|&i| h[i].is_zero()) || 
        x.deg().iter().any(|(i, &d_i)| 
            self.fst_exc_vars.contains(i) || 
            (self.snd_exc_vars.contains(i) && d_i >= 2)
        )
    }

    fn forward(&self, v: &VertGen) -> Vec<(VertGen, R)> {
        todo!()
    }

    fn backward(&self, w: &VertGen) -> Vec<(VertGen, R)> {
        todo!()
    }
}

fn rem<R>(f: &EdgeRing<R>, p: &EdgeRing<R>, i: usize) -> EdgeRing<R>
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