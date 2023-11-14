use std::collections::{HashSet, HashMap};

use num_traits::{Zero, One};
use yui::{Ring, RingOps, IndexList};
use yui_homology::utils::make_matrix_async;
use yui::lc::LinComb;
use yui_matrix::sparse::Trans;
use yui::poly::Mono;
use yui::bitseq::BitSeq;

use crate::kr::base::sign_between;

use super::base::{BaseMono, BasePoly, VertGen};
use super::data::KRCubeData;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    v_coords: BitSeq,
    edge_polys: HashMap<usize, BasePoly<R>>,
    exc_dirs: HashSet<usize>,
    fst_exc_vars: HashSet<usize>,
    snd_exc_vars: HashSet<usize>,
    remain_vars: HashSet<usize>,
    process: Vec<Process<R>>,
}

impl<R> KRHorExcl<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    fn new(v_coords: BitSeq, edge_polys: HashMap<usize, BasePoly<R>>) -> Self { 
        let n = v_coords.len();
        Self { 
            v_coords,
            edge_polys,
            exc_dirs: HashSet::new(), 
            fst_exc_vars: HashSet::new(), 
            snd_exc_vars: HashSet::new(),
            remain_vars: (0..n).collect(),
            process: vec![], 
        }
    }

    pub fn from(data: &KRCubeData<R>, v_coords: BitSeq, level: usize) -> Self { 
        assert!(level <= 2);

        let n = data.dim();
        let edge_polys = (0..n).map(|i| 
            (i, data.hor_edge_poly(v_coords, i))
        ).collect();

        let mut res = Self::new(v_coords, edge_polys);

        for d in 1..=level { 
            res.excl_all(d);
        }

        res
    }

    pub fn v_coords(&self) -> BitSeq { 
        self.v_coords
    }

    fn excl_all(&mut self, deg: usize) {
        while let Some((i, k)) = self.find_excl_var(deg) { 
            self.perform_excl(deg, i, k);
        }
    }

    fn find_excl_var(&self, deg: usize) -> Option<(usize, usize)> { 
        let cands = self.edge_polys.keys().filter_map(|&i| { 
            self.find_excl_var_for(deg, i).map(|k| (i, k))
        });

        cands.max_by(|(i1, _), (i2, _)| { 
            let p1 = self.edge_polys[&i1].ngens();
            let p2 = self.edge_polys[&i2].ngens();
            // prefer smaller poly.
            Ord::cmp(&p1, &p2).reverse().then(
                // prefer smaller index.
                Ord::cmp(&i1, &i2).reverse()
            )
        })
    }

    fn find_excl_var_for(&self, deg: usize, i: usize) -> Option<usize> { 
        let p = &self.edge_polys[&i];

        let cands = p.iter().filter_map(|(x, a)| {
            // term must be univar: a * (x_k)^d
            if let Some(k) = x.deg().min_index() { 
                if x.total_deg() == deg && x.deg_for(k) == deg { 
                    return Some((k, a))
                }
            }
            None
        });
        
        // choose best candidate
        cands.max_by(|(k1, a1), (k2, a2)| { 
            // prefer coeff ±1
            Ord::cmp(&a1.is_pm_one(), &a2.is_pm_one()).then( 
                // prefer smaller index
                Ord::cmp(&k1, &k2).reverse() 
            )
        }).map(|(k, _)| k)
    }

    fn perform_excl(&mut self, deg: usize, i: usize, k: usize) {
        // take divisor and edge-polys.
        let p = self.edge_polys.remove(&i).unwrap();
        let edge_polys = std::mem::take(&mut self.edge_polys);

        // update edge-polys.
        self.edge_polys = edge_polys.clone().into_iter().map(|(j, f)| {
            let f_rem = rem(f, &p, k);
            (j, f_rem)
        }).collect();

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
                    self.remain_vars.remove(l);
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

    pub fn reduce_gens(&self, gens: &Vec<VertGen>) -> Vec<VertGen> {
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

    pub fn forward(&self, z: &LinComb<VertGen, R>) -> LinComb<VertGen, R> {
        // keep only non-vanishing gens. 
        let z = z.filter_gens(|v| 
            !self.should_vanish(v)
        );

        // reduce monomials. 
        if z.iter().any(|(v, _)| self.should_reduce(&v.2)) { 
            let z = combine(z);
            let res = self.forward_reduce(z);
            decombine(res)
        } else { 
            z
        }

    }

    fn forward_reduce(&self, z: LinComb<VertGen, BasePoly<R>>) -> LinComb<VertGen, BasePoly<R>> { 
        let res = self.process.iter().fold(z, |z, proc| { 
            let (p, k) = proc.divisor();
            z.into_map_coeffs::<BasePoly<R>, _>(|f| 
                rem(f, p, k)
            )
        });

        // repeat if should-reduce monomials remain.
        if res.iter().any(|(_, f)| f.iter().any(|(x, _)| self.should_reduce(x))) { 
            self.forward_reduce(res)
        } else { 
            res
        }
    }

    fn forward_x(&self, v: &VertGen) -> LinComb<VertGen, R> {
        self.forward(&LinComb::from(v.clone()))
    }

    pub fn backward(&self, z: &LinComb<VertGen, R>, is_cycle: bool) -> LinComb<VertGen, R> {
        let init = combine(z.clone());
        let l = self.process.len();
        
        let res = if l > 0 { 
            self.backward_itr(init, l - 1, is_cycle)
        } else { 
            init
        };

        decombine(res)
    }

    fn backward_itr(&self, z: LinComb<VertGen, BasePoly<R>>, step: usize, is_cycle: bool) -> LinComb<VertGen, BasePoly<R>> {
        let res = self.backward_step(z, step, is_cycle);
        if step > 0 { 
            self.backward_itr(res, step - 1, is_cycle)
        } else { 
            res
        }
    }

    fn backward_step(&self, z: LinComb<VertGen, BasePoly<R>>, step: usize, is_cycle: bool) -> LinComb<VertGen, BasePoly<R>> {
        let d = &self.process[step];
        let i = d.dir;
        let (p, k) = d.divisor();

        //              i-th
        //         z . . . . . > z0
        //         |             |
        // d mod p |             | d
        //         V             V
        //        x1 . . . > x0, y0 ~> w = (x0 + y0)/p
        //

        let x0 = if !is_cycle {
            let x1 = self.d(&z, step, true);
            self.send_back(&x1, i)
        } else { 
            LinComb::<_, BasePoly<R>>::zero()
        };

        let z0 = self.send_back(&z, i);
        let y0 = self.d(&z0, step, false);

        let w = (x0 + y0).into_map_coeffs::<BasePoly<R>, _>(|f| 
            div(f, p, k)
        );
        
        z + w
    }

    fn backward_x(&self, w: &VertGen) -> LinComb<VertGen, R> {
        let z = LinComb::from(w.clone());
        self.backward(&z, false)
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
                    rem(f * g, p, k) * e
                } else { 
                    f * g * e
                };
                (w, h)
            }).collect::<F<R>>()
        }).sum()
    }

    pub fn trans_for(&self, from: &IndexList<VertGen>, to: &IndexList<VertGen>) -> Trans<R> {
        let fwd = make_matrix_async(from, to, |x| self.forward_x(x));
        let bwd = make_matrix_async(to, from, |x| self.backward_x(x));
        Trans::new(fwd, bwd)
    }

    pub fn diff_red(&self, z: &LinComb<VertGen, R>) -> LinComb<VertGen, R> { 
        z.apply(|x| self.diff_red_x(x))
    }

    fn diff_red_x(&self, e: &VertGen) -> LinComb<VertGen, R> { 
        let (h0, v0) = (e.0, e.1);
        let x0 = &e.2;

        self.edge_polys.iter().filter(|(&i, _)|
            h0[i].is_zero()
        ).flat_map(|(&i, f)| {

            let h1 = h0.edit(|b| b.set_1(i));
            let e = R::from_sign( sign_between(h0, h1) );
            let p = BasePoly::from((x0.clone(), e));
            let g = f * p;

            // expand as lin-comb
            let w = g.into_iter().map(|(x1, r)|
                (VertGen(h1, v0, x1), r)
            ).collect();

            self.forward(&w).into_iter()

        }).collect()
    }
}

// convert LinComb<VertGen, R> -> LinComb<VertGen, EdgeRing<R>> 
fn combine<R>(z: LinComb<VertGen, R>) -> LinComb<VertGen, BasePoly<R>>
where R: Ring, for<'x> &'x R: RingOps<R> {
    z.into_iter().map(|(v, r)| {
        let w = VertGen(v.0, v.1, BaseMono::one());
        let p = BasePoly::from((v.2, r));
        (w, p)
    }).collect()
}

// convert LinComb<VertGen, EdgeRing<R>> -> LinComb<VertGen, R> 
fn decombine<R>(z: LinComb<VertGen, BasePoly<R>>) -> LinComb<VertGen, R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    debug_assert!(z.iter().all(|(v, _)| v.2.is_one()));

    z.into_iter().flat_map(|(v, p)| { 
        p.into_iter().map(move |(x, a)| {
            let v = VertGen(v.0, v.1, x);
            (v, a)
        })
    }).collect()
}

fn div_rem<R>(f: BasePoly<R>, g: &BasePoly<R>, k: usize) -> (BasePoly<R>, BasePoly<R>)
where R: Ring, for<'x> &'x R: RingOps<R> {
    let (e0, a0) = g.lead_term_for(k).unwrap();

    assert!(e0.deg_for(k) > 0);
    assert!(a0.is_unit());

    let a0_inv = a0.inv().unwrap();

    let mut q = BasePoly::zero();
    let mut r = f;
    
    while let Some((e1, a1)) = r.lead_term_for(k) { 
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

fn div<R>(f: BasePoly<R>, p: &BasePoly<R>, k: usize) -> BasePoly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    let (q, r) = div_rem(f, p, k);
    debug_assert!(r.is_zero());
    q
}

fn rem<R>(f: BasePoly<R>, p: &BasePoly<R>, k: usize) -> BasePoly<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    div_rem(f, p, k).1
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use yui_link::Link;
    use yui_matrix::sparse::MatType;
    use yui::poly::MultiVar;
    use yui::Ratio;
    use yui::macros::map;
    use yui::bitseq::BitSeq;

    use crate::kr::data::KRCubeData;
    use crate::kr::hor_cube::KRHorCube;

    use super::*;

    type R = Ratio<i64>;
    type P = BasePoly<R>;

    fn vars(l: usize) -> Vec<P> {
        (0..l).map(P::variable).collect_vec()
    }

    fn vgen<const N: usize, const M: usize>(h: [usize; N], v: [usize; N], m: [usize; M]) -> VertGen {
        VertGen(BitSeq::from(h), BitSeq::from(v), MultiVar::from(m))
    }

    #[test]
    fn init() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let excl = KRHorExcl::from(&data, v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let excl = KRHorExcl::from(&data, v, 0);

        assert_eq!(excl.find_excl_var_for(1, 0), Some(1));
        assert_eq!(excl.find_excl_var_for(1, 1), Some(0));
        assert_eq!(excl.find_excl_var_for(1, 2), None);
        
        assert_eq!(excl.find_excl_var_for(2, 0), None);
        assert_eq!(excl.find_excl_var_for(2, 1), None);
        assert_eq!(excl.find_excl_var_for(1, 2), None);
    }

    #[test]
    fn perform_excl() { 
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

        // replaces x1 -> x2
        excl.perform_excl(1, 0, 1);

        assert_eq!(excl.find_excl_var_for(1, 1), Some(0));
        assert_eq!(excl.find_excl_var_for(1, 2), None);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

        // replaces x1 -> x2
        excl.perform_excl(1, 0, 1);

        assert_eq!(excl.find_excl_var_for(2, 1), None);
        assert_eq!(excl.find_excl_var_for(2, 2), Some(2));

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

        let data = KRCubeData::<R>::new(&l, 0);
        let excl = KRHorExcl::from(&data, v, 0);
        let cube = KRHorCube::new(Arc::new(data), v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

        excl.perform_excl(1, 0, 1);

        assert!( excl.should_vanish(&vgen([0,1,0], [0,0,1], [1,2,3]))); // h[0] = 0
        assert!(!excl.should_vanish(&vgen([1,0,0], [0,0,1], [1,2,3])));
    }

    #[test]
    fn should_reduce_before() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let excl = KRHorExcl::from(&data, v, 0);
        let cube = KRHorCube::new(Arc::new(data), v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

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

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);
        let cube = KRHorCube::new(Arc::new(data), v, 0);

        let g = (0..=3).map(|i| 
            cube.gens(i)
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

    #[test]
    fn forward() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

        excl.perform_excl(1, 0, 1); // x1 -> x2

        assert_eq!(excl.forward_x(&vgen([0,0,0], [0,0,1], [0,0,0])), LinComb::zero()); // vanish
        assert_eq!(excl.forward_x(&vgen([1,0,0], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,0,0], [0,0,1], [0,0,0]), R::one()) // 1: id
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,0], [0,0,1], [1,0,0])), LinComb::from_iter([
            (vgen([1,0,0], [0,0,1], [1,0,0]), R::one()) // x0: id
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,0], [0,0,1], [0,1,0])), LinComb::from_iter([
            (vgen([1,0,0], [0,0,1], [0,0,1]), R::one()) // x1 -> x2
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,0], [0,0,1], [0,0,1])), LinComb::from_iter([
            (vgen([1,0,0], [0,0,1], [0,0,1]), R::one()) // x2: id
        ]));
    }

    #[test]
    fn forward_2() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);

        excl.perform_excl(1, 0, 1); // x1 -> x2
        excl.perform_excl(2, 2, 2); // x2 * x2 -> x0 * x2

        assert_eq!(excl.forward_x(&vgen([0,1,1], [0,0,1], [0,0,0])), LinComb::zero()); // vanish
        assert_eq!(excl.forward_x(&vgen([1,1,0], [0,0,1], [0,0,0])), LinComb::zero()); // vanish

        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [0,0,0]), R::one()) // 1: id
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [1,0,0])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [1,0,0]), R::one()) // x1: id
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [0,1,0])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [0,0,1]), R::one()) // x1 -> x2
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [0,0,1])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [0,0,1]), R::one()) // x2: id
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [0,0,2])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [1,0,1]), R::one()) // x2^2 -> x0 x2
        ]));
        assert_eq!(excl.forward_x(&vgen([1,0,1], [0,0,1], [1,1,1])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [2,0,1]), R::one()) // x0x1x2 -> x0^2 x2
        ]));
    }

    #[test]
    fn forward_3() { 
        type R = Ratio<i64>;
        type P = BasePoly<R>;

        let h = BitSeq::from([1,1,1,1]);
        let v = BitSeq::from([1,1,1,1]);

        let x = |i| P::from(MultiVar::from((i, 1)));
        let edge_polys = map! { 
            0 => x(0) * x(0) - x(1) * x(1),
            1 => x(2) * x(2) - x(0) * x(3)
        };
        let mut excl = KRHorExcl::new(v, edge_polys);

        excl.perform_excl(2, 0, 0); // x0^2 -> x1^2
        excl.perform_excl(2, 1, 2); // x2^2 -> x0x3

        let x = VertGen(h, v, MultiVar::from([0,0,4,0])); // x2^4

        assert!(!excl.should_vanish(&x));
        assert!( excl.should_reduce(&x.2));

        let x = LinComb::from(x);
        let y = excl.forward(&x); // x2^4 -> (x0x3)^2 -> x1^2 x3^2

        assert_eq!(y, LinComb::from(VertGen(h, v, MultiVar::from([0,2,0,2]))))
    }

    #[test]
    fn backward() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);
        
        excl.perform_excl(1, 0, 1); // x1 -> x2

        assert_eq!(excl.backward_x(&vgen([1,0,0], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,0,0], [0,0,1], [0,0,0]),  R::one()), //  id at [1,0,0]
            (vgen([0,0,1], [0,0,1], [0,0,1]), -R::one())  // -x2 at [0,0,1]
        ]));

        assert_eq!(excl.backward_x(&vgen([1,1,0], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,1,0], [0,0,1], [0,0,0]),  R::one()), //  id at [1,1,0]
            (vgen([0,1,1], [0,0,1], [0,0,1]),  R::one())  //  x2 at [0,1,1]
        ]));

        assert_eq!(excl.backward_x(&vgen([1,0,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [0,0,0]),  R::one()), //  id at [1,0,1]
        ]));

        assert_eq!(excl.backward_x(&vgen([1,1,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,1,1], [0,0,1], [0,0,0]),  R::one()), //  id at [1,1,1]
        ]));
    }

    #[test]
    fn backward_2() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);
        
        excl.perform_excl(1, 0, 1); // x1 -> x2
        excl.perform_excl(1, 1, 0); // x0 -> x2

        assert_eq!(excl.backward_x(&vgen([1,1,0], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,1,0], [0,0,1], [0,0,0]),  R::one()), //  id at [1,1,0]
            (vgen([1,0,1], [0,0,1], [0,0,1]), -R::one()), // -x2 at [1,0,1]
            (vgen([0,1,1], [0,0,1], [0,0,1]),  R::one()), //  x2 at [0,1,1]
        ]));
        assert_eq!(excl.backward_x(&vgen([1,1,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,1,1], [0,0,1], [0,0,0]), R::one())
        ]));
    }

    #[test]
    fn backward_3() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);
        
        excl.perform_excl(1, 0, 1); // x1 -> x2
        excl.perform_excl(2, 2, 2); // x2^2 -> x0 x2

        assert_eq!(excl.backward_x(&vgen([1,0,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,0,1], [0,0,1], [0,0,0]), R::one())
        ]));
        assert_eq!(excl.backward_x(&vgen([1,1,1], [0,0,1], [0,0,0])), LinComb::from_iter([
            (vgen([1,1,1], [0,0,1], [0,0,0]), R::one())
        ]));
    }

    #[test]
    fn trans() { 
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,1]);

        let data = KRCubeData::<R>::new(&l, 0);
        let mut excl = KRHorExcl::from(&data, v, 0);
        let cube = KRHorCube::new(Arc::new(data), v, 0);

        let gens = cube.gens(2);

        let trans = |excl: &KRHorExcl<R>| {
            let from = IndexList::from_iter( gens.clone() );
            let to = IndexList::from_iter( excl.reduce_gens(&gens) );
            excl.trans_for(&from, &to)
        };

        let t0 = trans(&excl);

        assert_eq!(t0.forward_mat().shape(), (26, 26));
        assert!(t0.forward_mat().is_id());
        assert!(t0.backward_mat().is_id());

        excl.perform_excl(1, 0, 1); // x1 -> x2

        let t1 = trans(&excl);

        assert_eq!(t1.forward_mat().shape(), (7, 26));
        assert!((t1.forward_mat() * t1.backward_mat()).is_id());

        excl.perform_excl(1, 1, 0); // x0 -> x2

        let t2 = trans(&excl);

        assert_eq!(t2.forward_mat().shape(), (1, 26));
        assert!((t2.forward_mat() * t2.backward_mat()).is_id());
    }
}