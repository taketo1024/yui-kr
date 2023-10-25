use std::collections::BTreeMap;
use std::rc::Rc;

use itertools::Itertools;
use yui_core::{Ring, RingOps};
use yui_homology::XChainComplex;
use yui_polynomial::MultiDeg;
use yui_utils::bitseq::{BitSeq, Bit};

use super::base::{EdgeRing, MonGen, VertGen};
use super::data::KRCubeData;

pub(crate) type KRHorComplex<R> = XChainComplex<VertGen, R>;

pub(crate) struct KRHorCube<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    data: Rc<KRCubeData<R>>,
    v_coords: BitSeq,
    q_slice: isize,
} 

impl<R> KRHorCube<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(data: Rc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self {
        assert_eq!(v_coords.len(), data.dim());
        Self {
            data,
            v_coords,
            q_slice
        }
    }

    fn mon_deg(&self, h_coords: BitSeq) -> isize { 
        let i0 = self.data.root_grad().0;
        let i  = self.data.grad_at(h_coords, self.v_coords).0; // <= i0
        let h = h_coords.weight() as isize;

        self.q_slice + h + (i0 - i) / 2
    }

    fn make_mons(tot_deg: usize, n: usize) -> Vec<MonGen> { 
        fn gen_iter(tot_deg: usize, n: usize, res: &mut Vec<BTreeMap<usize, usize>>, prev: BTreeMap<usize, usize>, i: usize, rem: usize) {
            if i < n - 1 { 
                for d_i in (0..=rem).rev() { 
                    let mut curr = prev.clone();
                    curr.insert(i, d_i);
                    gen_iter(tot_deg, n, res, curr, i + 1, rem - d_i)
                }
            } else { 
                let mut curr = prev;
                curr.insert(i, rem);
                res.push(curr);
            }
        }

        let mut res = vec![];
        let prev = BTreeMap::new();

        gen_iter(tot_deg, n, &mut res, prev, 0, tot_deg);

        res.into_iter().map(|d| {
            let mdeg = MultiDeg::new(d);
            MonGen::from(mdeg)
        }).collect()
    }

    fn vert_gens(&self, h_coords: BitSeq) -> Vec<VertGen> {
        let deg = self.mon_deg(h_coords);
        if deg < 0 { 
            return vec![]
        }

        let deg = deg as usize;
        let n = self.data.dim();
        let gens = Self::make_mons(deg, n);

        gens.into_iter().map(|x| 
            VertGen(h_coords, self.v_coords, x)
        ).collect()
    }

    fn gens(&self, i: usize) -> Vec<VertGen> {
        self.data.verts(i).into_iter().flat_map(|v| 
            self.vert_gens(v)
        ).collect()
    }

    fn edge_poly(&self, from: BitSeq, to: BitSeq) -> EdgeRing<R> {
        use Bit::{Bit0, Bit1};
        assert_eq!(to.weight() - from.weight(), 1);

        let n = from.len();
        let i = (0..n).find(|&i| from[i] != to[i]).unwrap();

        let sign = self.data.x_signs()[i];
        let v = self.v_coords[i];
        let p = self.data.x_poly(i);

        let a = match (sign.is_positive(), v) {
            (true, Bit0) | (false, Bit1) => &p.x_ac * &p.x_bc,
            (true, Bit1) | (false, Bit0) => p.x_ac.clone()
        };

        a
    }

    fn differentiate(&self, e: &VertGen) -> Vec<(VertGen, R)> { 
        let VertGen(h, v, x) = e;
        let h0 = h.clone();
        let v = v.clone();
        
        self.data.targets(h0).into_iter().flat_map(|h1| { 
            let s = self.data.edge_sign(h0, h1);
            let e = EdgeRing::from_sign(s);
            let x = EdgeRing::from(x.clone());
            let p = self.edge_poly(h0, h1);
            let q = e * x * p; // result polynomial

            q.into_iter().map(move |(x, r)| 
                (VertGen(h1, v, x), r)
            )
        }).collect_vec()
    }

    pub fn as_complex(self) -> KRHorComplex<R> {
        let n = self.data.dim() as isize;
        let range = 0..=n;
        
        XChainComplex::new(range, 1, 
            |i| {
                self.gens(i as usize)
            },
            |_i, e| {
                self.differentiate(e)
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use num_traits::One;
    use yui_homology::ChainComplexTrait;
    use yui_ratio::Ratio;
    use yui_link::Link;
    use yui_utils::map;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

    fn make_cube(l: &Link, v: BitSeq, q: isize) -> KRHorCube<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Rc::new(data);
        let cube = KRHorCube::new(rc, v, q);
        cube
    }

    #[test]
    fn gen_mons() { 
        let tot = 5;
        let n = 3;
        let mons = KRHorCube::<R>::make_mons(tot, n);
        assert_eq!(mons.len(), 21); // (1,2,2) <-> *|**|**
    }

    #[test]
    fn mon_deg() { 
        let l = Link::trefoil();

        let v = BitSeq::from_iter([0,0,0]);
        let h = BitSeq::from_iter([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 0);

        let v = BitSeq::from_iter([0,0,0]);
        let h = BitSeq::from_iter([0,0,0]);
        let q = 1;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 1);

        let v = BitSeq::from_iter([1,0,0]);
        let h = BitSeq::from_iter([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 0);

        let v = BitSeq::from_iter([0,0,0]);
        let h = BitSeq::from_iter([1,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 1);

        let v = BitSeq::from_iter([1,0,0]);
        let h = BitSeq::from_iter([1,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 2);
    }

    #[test]
    fn edge_poly() { 
        let x = (0..3).map(|i| P::variable(i)).collect_vec();

        // p: neg, v: 0, h: 0 -> 1
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from_iter([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h0 = BitSeq::from_iter([0,0,0]);
        let h1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h0, h1); // x_ac
        assert_eq!(p, -&x[1] + &x[2]);

        // p: neg, v: 1, h: 0 -> 1
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h0 = BitSeq::from_iter([0,0,0]);
        let h1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h0, h1); // x_ac * x_bc
        assert_eq!(p, (-&x[1] + &x[2]) * &x[0]);

        // p: pos, v: 0, h: 0 -> 1
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]).mirror(); // trefoil
        let v = BitSeq::from_iter([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h0 = BitSeq::from_iter([0,0,0]);
        let h1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h0, h1); // x_ac * x_bc
        assert_eq!(p, (-&x[1] + &x[2]) * &x[0]);

        // p: pos, v: 1, h: 0 -> 1
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h0 = BitSeq::from_iter([0,0,0]);
        let h1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h0, h1); // x_ac
        assert_eq!(p, -&x[1] + &x[2]);
    }

    #[test]
    fn differentiate() { 
        let one = MonGen::one();
        let x = (0..3).map(|i| MonGen::from(MultiDeg::from((i, 1)))).collect_vec();

        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from_iter([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h = BitSeq::from_iter([0,0,0]);
        let z = VertGen(h, v, one.clone());
        let ys = cube.differentiate(&z);

        assert_eq!(ys.into_iter().collect::<HashMap<_, _>>(), map!{
            VertGen(BitSeq::from_iter([1,0,0]), v, x[1].clone()) => -R::one(),
            VertGen(BitSeq::from_iter([1,0,0]), v, x[2].clone()) =>  R::one(),
            VertGen(BitSeq::from_iter([0,1,0]), v, x[2].clone()) => -R::one(),
            VertGen(BitSeq::from_iter([0,1,0]), v, x[0].clone()) =>  R::one(),
            VertGen(BitSeq::from_iter([0,0,1]), v, x[0].clone()) => -R::one(),
            VertGen(BitSeq::from_iter([0,0,1]), v, x[1].clone()) =>  R::one()
        });

        let h = BitSeq::from_iter([0,1,0]);
        let z = VertGen(h, v, one.clone());
        let ys = cube.differentiate(&z);

        assert_eq!(ys.into_iter().collect::<HashMap<_, _>>(), map! {
            VertGen(BitSeq::from_iter([1,1,0]), v, x[1].clone()) => -R::one(),
            VertGen(BitSeq::from_iter([1,1,0]), v, x[2].clone()) =>  R::one(),
            VertGen(BitSeq::from_iter([0,1,1]), v, x[0].clone()) =>  R::one(),
            VertGen(BitSeq::from_iter([0,1,1]), v, x[1].clone()) => -R::one()
        });

        let h = BitSeq::from_iter([1,1,1]);
        let z = VertGen(h, v, one.clone());
        let ys = cube.differentiate(&z);

        assert_eq!(ys, vec![]);
    }

    #[test]
    fn vert_gens() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let h = BitSeq::from_iter([0, 0, 0]);
        let gens = cube.vert_gens(h);

        assert_eq!(gens.len(), 1);
        assert!(gens.iter().all(|x| x.0 == h));

        let h = BitSeq::from_iter([1, 0, 0]);
        let gens = cube.vert_gens(h);

        assert_eq!(gens.len(), 3);
        assert!(gens.iter().all(|x| x.0 == h));
    }

    #[test]
    fn gens() {
        let l = Link::trefoil();
        let v = BitSeq::from_iter([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let gens = cube.gens(0);
        assert_eq!(gens.len(), 1);

        let gens = cube.gens(1);
        assert_eq!(gens.len(), 9);
    }

    #[test]
    fn as_complex() { 
        let l = Link::trefoil();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);
        let c = cube.as_complex();

        assert_eq!(c.rank(0), 1);
        assert_eq!(c.rank(1), 12);
        assert_eq!(c.rank(2), 26);
        assert_eq!(c.rank(3), 15);

        c.check_d_all();

        let v = BitSeq::from_iter([1,0,0]);
        let q = 1;
        let cube = make_cube(&l.mirror(), v, q);
        let c = cube.as_complex();
        
        assert_eq!(c.rank(0), 6);
        assert_eq!(c.rank(1), 40);
        assert_eq!(c.rank(2), 70);
        assert_eq!(c.rank(3), 36);
        
        c.check_d_all();
    }

    #[test]
    fn homology() { 
        let l = Link::trefoil();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let c = cube.as_complex();
        let h = c.homology(false);

        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 1);
        assert_eq!(h[3].rank(), 1);

        let l = Link::trefoil().mirror();
        let v = BitSeq::from_iter([1,0,0]);
        let q = 1;
        let cube = make_cube(&l, v, q);
        let c = cube.as_complex();
        let h = c.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }
}