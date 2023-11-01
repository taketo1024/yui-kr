use std::collections::{HashSet, HashMap};
use std::hash::Hash;

use itertools::Itertools;
use num_traits::{Zero, One};
use yui_core::{Ring, RingOps, IndexList};
use yui_homology::XModStr;
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{Trans, SpMat};
use yui_polynomial::PolyGen;
use crate::kr::base::sign_between;

use super::base::{Poly, VertGen, Mono};
use super::hor_cube::KRHorCube;

struct Process<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    dir: usize,
    var: usize,
    divisor: Poly<R>,
    edge_polys: HashMap<usize, Poly<R>>
}

impl<R> Process<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn divisor(&self) -> (&Poly<R>, usize) { 
        (&self.divisor, self.var)
    }
}

pub(crate) struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    level: usize,
    edge_polys: HashMap<usize, Poly<R>>,
    exc_dirs: HashSet<usize>,
    fst_exc_vars: HashSet<usize>,
    snd_exc_vars: HashSet<usize>,
    remain_vars: HashSet<usize>,
    process: Vec<Process<R>>,
}

impl<R> KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(n: usize, edge_polys: HashMap<usize, Poly<R>>, level: usize) -> Self { 
        assert!(level <= 2);
        Self { 
            level,
            edge_polys,
            exc_dirs: HashSet::new(), 
            fst_exc_vars: HashSet::new(), 
            snd_exc_vars: HashSet::new(),
            remain_vars: (0..n).collect(),
            process: vec![], 
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
                if mdeg.of(k) == deg { 
                    return Some(k)
                }
            }
        }

        None
    }

    fn update_excl(&mut self, deg: usize, i: usize, k: usize) {
        debug_assert!(deg == 1 || deg == 2);

        // take divisor and preserve edge_polys.
        let p = self.edge_polys.remove(&k).unwrap();
        let edge_polys = self.edge_polys.clone();

        // update edge-polys.
        self.edge_polys = self.edge_polys.iter().map(|(&j, f)|
            (j, rem(f, &p, k))
        ).collect();

        // update other infos.
        self.exc_dirs.insert(i);
        if deg == 1 { 
            self.fst_exc_vars.insert(k);
        } else { 
            self.snd_exc_vars.insert(k);
        }

        // update indep vars.
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

        // insert new process.
        let d = Process {
            dir: i,
            var: k,
            divisor: p,
            edge_polys,
        };
        self.process.push(d);
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
        
        let f = Poly::from(v.2.clone());
        let f = self.process.iter().fold(f, |f, d| { 
            let (p, k) = d.divisor();
            rem(&f, p, k) // f mod p by x_k
        });

        f.into_iter().map(|(x, a)| {
            let v = VertGen(v.0, v.1, x);
            (v, a)
        }).collect()
    }

    fn backward(&self, w: &VertGen) -> Vec<(VertGen, R)> {
        type F<R> = LinComb<VertGen, Poly<R>>;

        // convert LinComb<VertGen, R> -> LinComb<VertGen, EdgeRing<R>> 
        let w0 = VertGen(w.0.clone(), w.1.clone(), Mono::one());
        let p = Poly::from(w.2.clone());
        let init = F::from((w0, p));

        let l = self.process.len();
        let res = if l > 0 { 
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

    fn backward_itr(&self, z: LinComb<VertGen, Poly<R>>, step: usize) -> LinComb<VertGen, Poly<R>> {
        let res = self.backward_step(z, step);
        if step > 0 { 
            self.backward_itr(res, step - 1)
        } else { 
            res
        }
    }

    fn backward_step(&self, z: LinComb<VertGen, Poly<R>>, step: usize) -> LinComb<VertGen, Poly<R>> {
        type F<R> = LinComb<VertGen, Poly<R>>;

        let d = &self.process[step];
        let i = d.dir;
        let (p, k) = d.divisor();

        //              i-th
        //         z . . . . . > z0
        //         |             |
        // d mod p |             | d
        //         V             V
        //        w1 . . . > w0, u0 ~> w = (w0 + u0)/p
        //

        let w1 = self.d(&z, step, true);
        let w0 = self.send_back(&w1, i);
        let z0 = self.send_back(&z, i);
        let u0 = self.d(&z0, step, false);

        let w = (w0 + u0).map_coeffs::<Poly<R>, _>(|f| 
            div(f, p, k)
        );
        
        z + w
    }

    fn send_back(&self, z: &LinComb<VertGen, Poly<R>>, dir: usize) -> LinComb<VertGen, Poly<R>> {
        let i = dir;
        z.map::<_, Poly<R>, _>(|v, f| { 
            debug_assert!(v.0[i].is_one());
            let u = VertGen(v.0.edit(|b| b.set_0(i)), v.1, v.2.clone());
            let e = R::from_sign( sign_between(u.0, v.0) );
            (u, f * e)
        })
    }

    fn d(&self, z: &LinComb<VertGen, Poly<R>>, step: usize, mod_p: bool) -> LinComb<VertGen, Poly<R>> { 
        type F<R> = LinComb<VertGen, Poly<R>>;
        
        let d = &self.process[step];
        let (p, k) = d.divisor();

        z.iter().map(|(v, f)| { 
            d.edge_polys.iter().filter(|(&i, _)|
                v.0[i].is_zero()
            ).map(|(&i, g)| {
                let w = VertGen(v.0.edit(|b| b.set_1(i)), v.1, v.2.clone());
                let e = R::from_sign( sign_between(v.0, w.0) );
                let h = if mod_p { 
                    rem(&(f * g), p, k) * e
                } else { 
                    f * g * e
                };
                (w, h)
            }).collect::<F<R>>()
        }).sum()
    }
}

fn div_rem<R>(f: &Poly<R>, p: &Poly<R>, k: usize) -> (Poly<R>, Poly<R>)
where R: Ring, for<'x> &'x R: RingOps<R> {
    todo!()
}

fn div<R>(f: &Poly<R>, p: &Poly<R>, k: usize) -> Poly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    let (q, r) = div_rem(f, p, k);
    debug_assert!(r.is_zero());
    q
}

fn rem<R>(f: &Poly<R>, p: &Poly<R>, k: usize) -> Poly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    div_rem(f, p, k).1
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