use std::collections::{HashSet, HashMap};
use std::hash::Hash;

use itertools::Itertools;
use num_traits::{Zero, One};
use yui_core::{Ring, RingOps, IndexList, Sign};
use yui_homology::XModStr;
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{Trans, SpMat};
use yui_polynomial::MonoGen;
use super::base::{EdgeRing, VertGen, MonGen};
use super::hor_cube::KRHorCube;

struct Divisor<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    dir: usize,
    var: usize,
    poly: EdgeRing<R>
}

impl<R> Divisor<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(dir: usize, var: usize, poly: EdgeRing<R>) -> Self { 
        Self { dir, var, poly }
    }

    fn get(&self) -> (usize, usize, &EdgeRing<R>) {
        (self.dir, self.var, &self.poly)
    }
}

pub(crate) struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    level: usize,
    edge_polys: Vec<HashMap<usize, EdgeRing<R>>>,
    divisors: Vec<Divisor<R>>,
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
            edge_polys: vec![edge_polys],
            divisors: vec![], 
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

    fn current_edge_polys(&self) -> &HashMap<usize, EdgeRing<R>> {
        &self.edge_polys.last().unwrap()
    }

    fn process_excl(&mut self, deg: usize) {
        assert!(deg == 1 || deg == 2);

        let current = self.current_edge_polys();
        let inds = current.keys().sorted().cloned().collect_vec();

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

        let current = self.current_edge_polys();
        let p = &current[&i];

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

        // insert new edge-polys
        let mut next = self.current_edge_polys().clone();
        let p = next.remove(&k).unwrap();
        let next = next.into_iter().map(|(j, f)|
            (j, rem(&f, &p, i))
        ).collect();
        self.edge_polys.push(next);

        // update other infos
        self.exc_dirs.insert(i);
        if deg == 1 { 
            self.fst_exc_vars.insert(k);
        } else { 
            self.snd_exc_vars.insert(k);
        }

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

        // insert new divisor
        let d = Divisor::new(i, k, p);
        self.divisors.push(d);
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
            if self.should_vanish(v) || self.should_reduce(v) { 
                None
            } else {
                Some(v)
            }
        ).collect()
    }

    fn should_vanish(&self, v: &VertGen) -> bool { 
        let h = &v.0;
        self.exc_dirs.iter().any(|&i| 
            h[i].is_zero()
        )
    }

    fn should_reduce(&self, v: &VertGen) -> bool { 
        let x = &v.2;
        x.deg().iter().any(|(k, &d)| 
            self.fst_exc_vars.contains(k) || 
            (self.snd_exc_vars.contains(k) && d >= 2)
        )
    }

    fn forward(&self, v: &VertGen) -> Vec<(VertGen, R)> {
        if self.should_vanish(v) { 
            return vec![]
        } 
        
        if !self.should_reduce(v) {
            return vec![(v.clone(), R::one())]
        }
        
        let f = EdgeRing::from(v.2.clone());
        let f = self.divisors.iter().fold(f, |f, d| { 
            let Divisor{var: k, poly: p, ..} = d;
            rem(&f, p, *k) // f mod p by x_k
        });

        f.into_iter().map(|(x, a)| {
            let v = VertGen(v.0, v.1, x);
            (v, a)
        }).collect()
    }

    fn backward(&self, w: &VertGen) -> Vec<(VertGen, R)> {
        // MEMO convert LinComb<VertGen, R> -> LinComb<VertGen, EdgeRing<R>> 
        type F<R> = LinComb<VertGen, EdgeRing<R>>;

        let w0 = VertGen(w.0.clone(), w.1.clone(), MonGen::one());
        let p = EdgeRing::from(w.2.clone());
        let init = F::from((w0, p));

        let l = self.divisors.len();
        let res = if l > 1 { 
            self.backward_itr(init, l - 1)
        } else { 
            init
        };

        res.into_iter().flat_map(|(v, p)| { 
            p.into_iter().map(move |(x, a)| {
                let v = VertGen(v.0.clone(), v.1.clone(), x);
                (v, a)
            })
        }).collect()
    }

    fn backward_itr(&self, res: LinComb<VertGen, EdgeRing<R>>, step: usize) -> LinComb<VertGen, EdgeRing<R>> {
        assert!(step > 0);
        type F<R> = LinComb<VertGen, EdgeRing<R>>;

        let (i, k, p) = self.divisors[step].get();
        let ep = &self.edge_polys[step];

        let res = res.into_iter().map(|(v, f)| { 
            let w1 = F::from(v.clone());
            let u1 = self.d(&v, step);
            let u2 = self.d(&v, step - 1);
            let e1 = R::from_sign(Sign::Pos);
            let e2 = R::from_sign(Sign::Pos);
            // let w2 = (u1 * e1 + u2 * e2).into_map_coeffs(||)
            w1
        }).sum();

        if step > 1 { 
            self.backward_itr(res, step - 1)
        } else { 
            res
        }
    }

    fn d(&self, v: &VertGen, step: usize) -> LinComb<VertGen, EdgeRing<R>> { 
        todo!()
    }
}

fn div_rem<R>(f: &EdgeRing<R>, p: &EdgeRing<R>, i: usize) -> (EdgeRing<R>, EdgeRing<R>)
where R: Ring, for<'x> &'x R: RingOps<R> {
    todo!()
}

fn div<R>(f: &EdgeRing<R>, p: &EdgeRing<R>, i: usize) -> EdgeRing<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    let (q, r) = div_rem(f, p, i);
    debug_assert!(r.is_zero());
    q
}

fn rem<R>(f: &EdgeRing<R>, p: &EdgeRing<R>, i: usize) -> EdgeRing<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    div_rem(f, p, i).1
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