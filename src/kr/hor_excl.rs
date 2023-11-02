use std::collections::{HashSet, HashMap};

use itertools::Itertools;
use num_traits::{Zero, One};
use yui_core::{Ring, RingOps, IndexList};
use yui_homology::{XModStr, make_matrix};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::Trans;
use yui_polynomial::Mono;

use super::base::{BaseMono, BasePoly, VertGen};
use super::data::sign_between;
use super::hor_cube::KRHorCube;

struct Process<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    dir: usize,
    var: usize,
    divisor: BasePoly<R>,
    edge_polys: HashMap<usize, BasePoly<R>>
}

impl<R> Process<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn divisor(&self) -> (&BasePoly<R>, usize) { 
        (&self.divisor, self.var)
    }
}

pub(crate) struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    edge_polys: HashMap<usize, BasePoly<R>>,
    exc_dirs: HashSet<usize>,
    fst_exc_vars: HashSet<usize>,
    snd_exc_vars: HashSet<usize>,
    remain_vars: HashSet<usize>,
    process: Vec<Process<R>>,
}

impl<R> KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(n: usize, edge_polys: HashMap<usize, BasePoly<R>>) -> Self { 
        Self { 
            edge_polys,
            exc_dirs: HashSet::new(), 
            fst_exc_vars: HashSet::new(), 
            snd_exc_vars: HashSet::new(),
            remain_vars: (0..n).collect(),
            process: vec![], 
        }
    }

    pub fn from(cube: &KRHorCube<R>, level: usize) -> Self { 
        assert!(level <= 2);

        let n = cube.data().dim();
        let edge_polys = (0..n).map(|i| 
            (i, cube.edge_poly_dir(i))
        ).collect();

        let mut res = Self::new(n, edge_polys);

        for d in 1..=level { 
            res.excl_all(d);
        }

        res
    }

    fn excl_all(&mut self, deg: usize) {
        assert!(deg == 1 || deg == 2);

        let inds = self.edge_polys.keys().sorted().cloned().collect_vec();

        for i in inds { 
            if let Some(k) = self.find_excl_var(deg, i) { 
                self.perform_excl(deg, i, k);
            }
        }
    }

    fn find_excl_var(&self, deg: usize, i: usize) -> Option<usize> { 
        assert!(deg == 1 || deg == 2);

        if self.remain_vars.is_empty() { 
            return None;
        }

        let p = &self.edge_polys[&i];

        // search for the term (x_k)^d in p.
        for &k in self.remain_vars.iter().sorted() {
            for (x, _) in p.iter() {
                if x.total_deg() == deg && x.deg_for(k) == deg { 
                    return Some(k)
                }
            }
        }

        None
    }

    fn perform_excl(&mut self, deg: usize, i: usize, k: usize) {
        debug_assert!(deg == 1 || deg == 2);

        // take divisor and preserve edge_polys.
        let p = self.edge_polys.remove(&i).unwrap();
        let edge_polys = self.edge_polys.clone();

        debug_assert_eq!(p.lead_term_for(k).0.total_deg(), deg);
        debug_assert_eq!(p.lead_term_for(k).0.deg_for(k),  deg);

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
        let from = v.gens();
        let to = self.reduce_gens(&from);

        let fwd = make_matrix(&from, &to, |x| self.forward(x));
        let bwd = make_matrix(&to, &from, |x| self.backward(x));

        Trans::new(fwd, bwd)
    }

    fn reduce_gens<'a>(&self, gens: &IndexList<VertGen>) -> IndexList<VertGen> {
        gens.iter().filter_map(|v| 
            if self.should_vanish(v) || self.should_reduce(&v.2) { 
                None
            } else {
                Some(v.clone())
            }
        ).collect()
    }

    fn should_vanish(&self, v: &VertGen) -> bool { 
        let h = &v.0;
        self.exc_dirs.iter().any(|&i| 
            h[i].is_zero()
        )
    }

    fn should_reduce(&self, x: &BaseMono) -> bool { 
        x.deg().iter().any(|(k, &d)| 
            self.fst_exc_vars.contains(k) || 
            (self.snd_exc_vars.contains(k) && d >= 2)
        )
    }

    fn forward(&self, v: &VertGen) -> Vec<(VertGen, R)> {
        if self.should_vanish(v) { 
            return vec![]
        } 
        
        if !self.should_reduce(&v.2) {
            return vec![(v.clone(), R::one())]
        }
        
        let f = BasePoly::from(v.2.clone());
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
        type F<R> = LinComb<VertGen, BasePoly<R>>;

        // convert LinComb<VertGen, R> -> LinComb<VertGen, EdgeRing<R>> 
        let w0 = VertGen(w.0.clone(), w.1.clone(), BaseMono::one());
        let p = BasePoly::from(w.2.clone());
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

    fn backward_itr(&self, z: LinComb<VertGen, BasePoly<R>>, step: usize) -> LinComb<VertGen, BasePoly<R>> {
        let res = self.backward_step(z, step);
        if step > 0 { 
            self.backward_itr(res, step - 1)
        } else { 
            res
        }
    }

    fn backward_step(&self, z: LinComb<VertGen, BasePoly<R>>, step: usize) -> LinComb<VertGen, BasePoly<R>> {
        type F<R> = LinComb<VertGen, BasePoly<R>>;

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

        let w = (w0 + u0).map_coeffs::<BasePoly<R>, _>(|f| 
            div(f, p, k)
        );
        
        z + w
    }

    fn send_back(&self, z: &LinComb<VertGen, BasePoly<R>>, dir: usize) -> LinComb<VertGen, BasePoly<R>> {
        let i = dir;
        z.map::<_, BasePoly<R>, _>(|v, f| { 
            debug_assert!(v.0[i].is_one());
            let u = VertGen(v.0.edit(|b| b.set_0(i)), v.1, v.2.clone());
            let e = R::from_sign( sign_between(u.0, v.0) );
            (u, f * e)
        })
    }

    fn d(&self, z: &LinComb<VertGen, BasePoly<R>>, step: usize, mod_p: bool) -> LinComb<VertGen, BasePoly<R>> { 
        type F<R> = LinComb<VertGen, BasePoly<R>>;
        
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

    #[cfg(test)]
    #[allow(dead_code)]
    fn print_current(&self) { 
        for (i, p) in self.edge_polys.iter().sorted_by_key(|(&i, _)| i) { 
            println!("{i}: {p}")
        }
    }
}

fn div_rem<R>(f: &BasePoly<R>, g: &BasePoly<R>, k: usize) -> (BasePoly<R>, BasePoly<R>)
where R: Ring, for<'x> &'x R: RingOps<R> {
    let mut q = BasePoly::zero();
    let mut r = f.clone();
    
    let (e0, a0) = g.lead_term_for(k);

    assert!(e0.deg_for(k) > 0);
    assert!(a0.is_unit()); // ±1

    let a0_inv = a0.inv().unwrap();

    while !r.is_zero() { 
        let (e1, a1) = r.lead_term_for(k);
        if !e0.divides(e1) {
            break
        }

        let b = a1 * &a0_inv;
        let x = BasePoly::from((e1 / e0, b));

        r -= &x * g;
        q += x;
    }
    
    (q, r)
}

fn div<R>(f: &BasePoly<R>, p: &BasePoly<R>, k: usize) -> BasePoly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    let (q, r) = div_rem(f, p, k);
    debug_assert!(r.is_zero());
    q
}

fn rem<R>(f: &BasePoly<R>, p: &BasePoly<R>, k: usize) -> BasePoly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    div_rem(f, p, k).1
}

#[cfg(test)]
#[allow(unused_imports)] // TODO remove later
mod tests {
    use std::rc::Rc;

    use yui_homology::{ChainComplexDisplay, DisplaySeq};
    use yui_link::Link;
    use yui_polynomial::MultiVar;
    use yui_ratio::Ratio;
    use yui_utils::{bitseq::BitSeq, map};

    use crate::kr::data::KRCubeData;

    use super::*;

    type R = Ratio<i64>;
    type P = BasePoly<R>;

    fn make_cube(l: &Link, v: BitSeq, q: isize) -> KRHorCube<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Rc::new(data);
        KRHorCube::new(rc, v, q)
    }

    fn vars(l: usize) -> Vec<P> {
        (0..l).map(|i| P::variable(i)).collect_vec()
    }

    fn vgen<const N: usize, const M: usize>(h: [usize; N], v: [usize; N], m: [usize; M]) -> VertGen {
        VertGen(BitSeq::from(h), BitSeq::from(v), MultiVar::from(m))
    }

    #[test]
    fn init() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let excl = KRHorExcl::from(&cube, 0);

        assert!(excl.exc_dirs.is_empty());
        assert!(excl.fst_exc_vars.is_empty());
        assert!(excl.snd_exc_vars.is_empty());
        assert_eq!(excl.remain_vars, [0,1,2].into());
        assert!(excl.process.is_empty());

        let xs = vars(3);
        let x0 = &xs[0];
        let x1 = &xs[1];
        let x2 = &xs[2];

        assert_eq!(excl.edge_polys, map!{ 
            0 => x1 - x2,
            1 => -x0 + x2,
            2 => x0 * x2 - x1 * x2
        });
    }

    #[test]
    fn find_excl_var() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let excl = KRHorExcl::from(&cube, 0);

        assert_eq!(excl.find_excl_var(1, 0), Some(1));
        assert_eq!(excl.find_excl_var(1, 1), Some(0));
        assert_eq!(excl.find_excl_var(1, 2), None);
        
        assert_eq!(excl.find_excl_var(2, 0), None);
        assert_eq!(excl.find_excl_var(2, 1), None);
        assert_eq!(excl.find_excl_var(1, 2), None);
    }

    #[test]
    fn perform_excl() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);

        // replaces x1 -> x2
        excl.perform_excl(1, 0, 1);

        let xs = vars(3);
        let x0 = &xs[0];
        let x1 = &xs[1];
        let x2 = &xs[2];

        assert_eq!(excl.edge_polys, map!{ 
            1 => -x0 + x2,
            2 => x0 * x2 - x2 * x2
        });

        assert_eq!(excl.exc_dirs, [0].into());
        assert_eq!(excl.fst_exc_vars, [1].into());
        assert_eq!(excl.snd_exc_vars, [].into());
        assert_eq!(excl.remain_vars, [0, 2].into());
        assert_eq!(excl.process.len(), 1);

        let proc = &excl.process[0];

        assert_eq!(proc.dir, 0);
        assert_eq!(proc.var, 1);
        assert_eq!(proc.divisor, x1 - x2);
        assert_eq!(proc.edge_polys, map!{ 
            1 => -x0 + x2,
            2 => x0 * x2 - x1 * x2
        });
    }

    #[test]
    fn perform_excl_2() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);

        // replaces x1 -> x2
        excl.perform_excl(1, 0, 1);

        assert_eq!(excl.find_excl_var(1, 1), Some(0));
        assert_eq!(excl.find_excl_var(1, 2), None);

        // replaces x0 -> x2
        excl.perform_excl(1, 1, 0);

        let xs = vars(3);
        let x0 = &xs[0];
        let x2 = &xs[2];

        assert_eq!(excl.edge_polys, map!{ 
            2 => P::zero()
        });

        assert_eq!(excl.exc_dirs, [0,1].into());
        assert_eq!(excl.fst_exc_vars, [0,1].into());
        assert_eq!(excl.snd_exc_vars, [].into());
        assert_eq!(excl.remain_vars, [2].into());
        assert_eq!(excl.process.len(), 2);

        let proc = &excl.process[1];
        
        assert_eq!(proc.dir, 1);
        assert_eq!(proc.var, 0);
        assert_eq!(proc.divisor, -x0 + x2);
        assert_eq!(proc.edge_polys, map!{ 
            2 => x0 * x2 - x2 * x2
        });
    }

    #[test]
    fn perform_excl_3() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);

        // replaces x1 -> x2
        excl.perform_excl(1, 0, 1);

        assert_eq!(excl.find_excl_var(2, 1), None);
        assert_eq!(excl.find_excl_var(2, 2), Some(2));

        // replaces x2 * x2 -> x0 * x2
        excl.perform_excl(2, 2, 2);

        let xs = vars(3);
        let x0 = &xs[0];
        let x2 = &xs[2];

        assert_eq!(excl.edge_polys, map!{ 
            1 => -x0 + x2
        });

        assert_eq!(excl.exc_dirs, [0,2].into());
        assert_eq!(excl.fst_exc_vars, [1].into());
        assert_eq!(excl.snd_exc_vars, [2].into());
        assert_eq!(excl.remain_vars, [].into()); // no indep vars.
        assert_eq!(excl.process.len(), 2);

        let proc = &excl.process[1];

        assert_eq!(proc.dir, 2);
        assert_eq!(proc.var, 2);
        assert_eq!(proc.divisor, x0 * x2 - x2 * x2);
        assert_eq!(proc.edge_polys, map!{ 
            1 => -x0 + x2
        });
    }

    #[test]
    fn should_vanish_before() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let excl = KRHorExcl::from(&cube, 0);

        assert!((0..=3).all(|i| 
            cube.gens(i).iter().all(|v| 
                !excl.should_vanish(v)
            )
        ));
    }

    #[test]
    fn should_vanish() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);
        excl.perform_excl(1, 0, 1);

        assert!( excl.should_vanish(&vgen([0,1,0], [0,0,1], [1,2,3]))); // h[0] = 0
        assert!(!excl.should_vanish(&vgen([1,0,0], [0,0,1], [1,2,3])));
    }

    #[test]
    fn should_reduce_before() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let excl = KRHorExcl::from(&cube, 0);

        assert!((0..=3).all(|i| 
            cube.gens(i).iter().all(|v| 
                !excl.should_reduce(&v.2)
            )
        ));
    }

    #[test]
    fn should_reduce() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);
        excl.perform_excl(1, 0, 1); // x_1 is reduced

        assert!(!excl.should_reduce(&MultiVar::from([0,0,0])));
        assert!(!excl.should_reduce(&MultiVar::from([1,0,0])));
        assert!( excl.should_reduce(&MultiVar::from([0,1,0])));
        assert!(!excl.should_reduce(&MultiVar::from([0,0,1])));
    }

    #[test]
    fn should_reduce_2() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);
        excl.perform_excl(1, 0, 1); // x_1 is reduced.
        excl.perform_excl(2, 2, 2); // (x_2)^2 is reduced.

        assert!(!excl.should_reduce(&MultiVar::from([0,0,0])));
        assert!(!excl.should_reduce(&MultiVar::from([1,0,0])));
        assert!( excl.should_reduce(&MultiVar::from([0,1,0])));
        assert!(!excl.should_reduce(&MultiVar::from([0,0,1])));
        assert!( excl.should_reduce(&MultiVar::from([0,0,2])));
    }

    #[test]
    fn reduce_gens() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let mut excl = KRHorExcl::from(&cube, 0);

        let g = (0..=3).map(|i| 
            IndexList::from_iter(cube.gens(i))
        ).collect_vec();

        assert_eq!(g[0].len(),  1);
        assert_eq!(g[1].len(), 12);
        assert_eq!(g[2].len(), 26);
        assert_eq!(g[3].len(), 15);

        excl.perform_excl(1, 0, 1); // x_1 is reduced.

        let rg = g.iter().map(|g| 
            excl.reduce_gens(g)
        ).collect_vec();

        assert_eq!(rg[0].len(), 0);
        assert_eq!(rg[1].len(), 2);
        assert_eq!(rg[2].len(), 7);
        assert_eq!(rg[3].len(), 5);

        excl.perform_excl(2, 2, 2); // (x_2)^2 is reduced.

        let rg = g.iter().map(|g| 
            excl.reduce_gens(g)
        ).collect_vec();

        assert_eq!(rg[0].len(), 0);
        assert_eq!(rg[1].len(), 0);
        assert_eq!(rg[2].len(), 2);
        assert_eq!(rg[3].len(), 2);
    }
}