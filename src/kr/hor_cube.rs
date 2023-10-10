use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::rc::Rc;

use itertools::{izip, Itertools};
use num_traits::Zero;
use yui_core::{Ring, RingOps};
use yui_homology::FreeChainComplex;
use yui_polynomial::MDegree;
use yui_utils::bitseq::{BitSeq, Bit};

use super::base::{EdgeRing, TripGrad, MonGen};
use super::data::KRCubeData;

struct KRHorCube<R>
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

    fn grad_shift_x(x_sign: i32, h: Bit, v: Bit) -> TripGrad { 
        use Bit::{Bit0, Bit1};
        if x_sign.is_positive() {
            match (h, v) {
                (Bit0, Bit0) => TripGrad(2, -2, -2),
                (Bit1, Bit0) => TripGrad(0,  0, -2),
                (Bit0, Bit1) => TripGrad(0, -2,  0),
                (Bit1, Bit1) => TripGrad(0,  0,  0)
            }
        } else { 
            match (h, v) {
                (Bit0, Bit0) => TripGrad( 0, -2, 0),
                (Bit1, Bit0) => TripGrad( 0,  0, 0),
                (Bit0, Bit1) => TripGrad( 0, -2, 2),
                (Bit1, Bit1) => TripGrad(-2,  0, 2)
            }
        }
    }

    fn grad_shift(x_signs: &Vec<i32>, h_coords: BitSeq, v_coords: BitSeq) -> TripGrad { 
        assert_eq!(x_signs.len(), h_coords.len());
        assert_eq!(x_signs.len(), v_coords.len());

        let zero = TripGrad::zero();
        let trip = izip!(
            x_signs.iter(), 
            h_coords.iter(), 
            v_coords.iter()
        );
        trip.fold(zero, |grad, (&e, h, v)| { 
            grad + Self::grad_shift_x(e, h, v)
        })
    }

    fn root_grad(&self) -> TripGrad { 
        let n = self.data.dim();
        let h0 = BitSeq::zeros(n);
        let v0 = BitSeq::zeros(n);
        Self::grad_shift(self.data.x_signs(), h0, v0)
    }

    fn grad_at(&self, h_coords: BitSeq) -> TripGrad { 
        Self::grad_shift(self.data.x_signs(), h_coords, self.v_coords)
    }

    fn mon_deg(&self, h_coords: BitSeq) -> isize { 
        self.q_slice 
        + (h_coords.weight() as isize)
        + (self.root_grad().0 - self.grad_at(h_coords).0) / 2
    }

    fn gen_mons(tot_deg: usize, n: usize) -> Vec<MonGen> { 
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
            let mdeg = MDegree::new(d);
            MonGen::from(mdeg)
        }).collect()
    }

    fn vertex_gens(&self, h_coords: BitSeq) -> Vec<MonGen> {
        let deg = self.mon_deg(h_coords);
        if deg < 0 { 
            return vec![]
        }

        let deg = deg as usize;
        let n = self.data.dim();

        Self::gen_mons(deg, n)
    }

    fn generators(&self, k: usize) -> Vec<MonGen> {
        let n = self.data.dim();

        // TODO cache
        let vs = BitSeq::generate(n).into_iter().filter(|v| v.weight() == k);

        vs.flat_map(|v| self.vertex_gens(v)).collect_vec()
    }

    // TODO cache
    fn edge_sign(&self, from: BitSeq, to: BitSeq) -> i32 { 
        todo!()
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

    // TODO cache
    fn targets(&self, from: BitSeq) -> Vec<BitSeq> { 
        let n = from.len();
        (0..n).filter(|&i| from[i].is_zero() ).map(|i| { 
            let mut t = from.clone();
            t.set_1(i);
            t
        }).collect()
    }

    fn differentiate(&self, from: BitSeq, x: MonGen) -> Vec<(BitSeq, EdgeRing<R>)> { 
        self.targets(from).into_iter().map(|to| { 
            let e = EdgeRing::from(self.edge_sign(from, to));
            let x = EdgeRing::from_term(x.clone(), R::one());
            let p = self.edge_poly(from, to);
            (to, e * x * p)
        }).collect()
    }

    fn as_complex(self) -> FreeChainComplex<MonGen, R, RangeInclusive<isize>> {
        let n = self.data.dim() as isize;
        let range = 0..=n;
        
        let self0 = Rc::new(self);
        let self1 = self0.clone();

        FreeChainComplex::new(range, 1, 
            move |i| {
                todo!()
            },
            move |x| { 
                todo!()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

    fn make_cube(l: &Link, v: BitSeq, q: isize) -> KRHorCube<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Rc::new(data);
        let cube = KRHorCube::new(rc, v, 0);
        cube
    }

    #[test]
    fn gen_mons() { 
        let tot = 5;
        let n = 3;
        let mons = KRHorCube::<R>::gen_mons(tot, n);
        assert_eq!(mons.len(), 21); // (1,2,2) <-> *|**|**
    }

    #[test]
    fn grad_shift() { 
        let signs = vec![1,1,-1,-1];
        let h = BitSeq::from_iter([0,1,0,1]);
        let v = BitSeq::from_iter([0,0,1,1]);
        let grad = KRHorCube::<R>::grad_shift(&signs, h, v);
        assert_eq!(grad, TripGrad(0,-4,0));
    }

    #[test]
    fn vert_grad() { 
        let l = Link::hopf_link(); // negative
        let v = BitSeq::from_iter([0,1]);
        let cube = make_cube(&l, v, 0);

        let grad0 = cube.root_grad();
        assert_eq!(grad0, TripGrad(0,-4,0));

        let h = BitSeq::from_iter([1,0]);
        let grad1 = cube.grad_at(h);
        assert_eq!(grad1, TripGrad(0,-2,2));
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
}