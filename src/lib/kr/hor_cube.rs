use std::sync::Arc;

use yui::{Ring, RingOps};
use yui_homology::{XChainComplex, Grid1, XModStr};
use yui::bitseq::BitSeq;

use super::base::{KRPoly, KRMono, KRGen, KRChain, sign_between};
use super::data::KRCubeData;

pub struct KRHorCube<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    data: Arc<KRCubeData<R>>,
    v_coords: BitSeq,
    q_slice: isize,
} 

impl<R> KRHorCube<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, v_coords: BitSeq, q_slice: isize) -> Self {
        assert_eq!(v_coords.len(), data.dim());
        Self {
            data,
            v_coords,
            q_slice
        }
    }

    pub fn dim(&self) -> usize { 
        self.data.dim()
    }

    pub fn data(&self) -> &KRCubeData<R> { 
        self.data.as_ref()
    }

    fn mon_deg(&self, h_coords: BitSeq) -> isize { 
        let i0 = self.data.root_grad().0;
        let i  = self.data.grad_at(h_coords, self.v_coords).0; // <= i0
        let h = h_coords.weight() as isize;

        self.q_slice + h + (i0 - i) / 2
    }

    pub fn vert_gens(&self, h_coords: BitSeq) -> Vec<KRGen> {
        let deg = self.mon_deg(h_coords);
        if deg < 0 { 
            return vec![]
        }

        let deg = deg as usize;
        let n = self.dim();
        let gens = KRMono::generate(n, deg);

        gens.into_iter().map(|x| 
            KRGen(h_coords, self.v_coords, x)
        ).collect()
    }

    pub fn gens(&self, i: usize) -> Vec<KRGen> {
        self.data.verts(i).into_iter().flat_map(|v| 
            self.vert_gens(v)
        ).collect()
    }

    pub fn edge_poly(&self, i: usize) -> KRPoly<R> {
        self.data.hor_edge_poly(self.v_coords, i)
    }

    pub fn d(&self, z: &KRChain<R>) -> KRChain<R> { 
        z.apply(|x| self.d_x(x))
    }

    fn d_x(&self, e: &KRGen) -> KRChain<R> { 
        let (h0, v0) = (e.0, e.1);
        let x0 = &e.2;
        let n = self.dim();

        (0..n).filter(|&i| 
            h0[i].is_zero()
        ).flat_map(|i| {
            let h1 = h0.edit(|b| b.set_1(i));
            let e = R::from_sign( sign_between(h0, h1) );
            let p = KRPoly::from( (x0.clone(), e ) );
            let f = self.edge_poly(i);
            let q = f * p;

            q.into_iter().map(move |(x1, r)| 
                (KRGen(h1, v0, x1), r)
            )
         }).collect()
    }

    pub fn into_complex(self) -> XChainComplex<KRGen, R> {
        let n = self.dim() as isize;
        let summands = Grid1::generate(0..=n, |i| { 
            let gens = self.gens(i as usize);
            XModStr::free(gens)
        });
        XChainComplex::new(summands, 1, move |_, z| {
            self.d(z)
        })
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};
    use yui_homology::{RModStr, ChainComplexCommon};
    use yui::Ratio;
    use yui_link::Link;
    use yui::util::macros::map;
    use super::*;

    type R = Ratio<i64>;
    type P = KRPoly<R>;

    fn make_cube(l: &Link, v: BitSeq, q: isize) -> KRHorCube<R> {
        let data = KRCubeData::<R>::new(l, 0);
        let rc = Arc::new(data);
        
        KRHorCube::new(rc, v, q)
    }

    #[test]
    fn mon_deg() { 
        let l = Link::trefoil();

        let v = BitSeq::from([0,0,0]);
        let h = BitSeq::from([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 0);

        let v = BitSeq::from([0,0,0]);
        let h = BitSeq::from([0,0,0]);
        let q = 1;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 1);

        let v = BitSeq::from([1,0,0]);
        let h = BitSeq::from([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 0);

        let v = BitSeq::from([0,0,0]);
        let h = BitSeq::from([1,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 1);

        let v = BitSeq::from([1,0,0]);
        let h = BitSeq::from([1,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let deg = cube.mon_deg(h);
        assert_eq!(deg, 2);
    }

    #[test]
    fn edge_poly() { 
        let x = P::variable;

        // p: neg, v: 0, h: 0 -> 1
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let p = cube.edge_poly(0); // x_ac
        assert_eq!(p, -&x(1) + &x(2));

        // p: neg, v: 1, h: 0 -> 1
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let p = cube.edge_poly(0); // x_ac * x_bc
        assert_eq!(p, (-&x(1) + &x(2)) * &x(0));

        // p: pos, v: 0, h: 0 -> 1
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]).mirror(); // trefoil
        let v = BitSeq::from([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let p = cube.edge_poly(0); // x_ac * x_bc
        assert_eq!(p, (-&x(1) + &x(2)) * &x(0));

        // p: pos, v: 1, h: 0 -> 1
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let p = cube.edge_poly(0); // x_ac
        assert_eq!(p, -&x(1) + &x(2));
    }

    #[test]
    fn differentiate() { 
        let one = KRMono::one();
        let x = |i| KRMono::from((i, 1));

        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let v = BitSeq::from([0,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let h = BitSeq::from([0,0,0]);
        let z = KRGen(h, v, one.clone());
        let ys = cube.d_x(&z);

        assert_eq!(ys, map!{
            KRGen(BitSeq::from([1,0,0]), v, x(1)) => -R::one(),
            KRGen(BitSeq::from([1,0,0]), v, x(2)) =>  R::one(),
            KRGen(BitSeq::from([0,1,0]), v, x(2)) => -R::one(),
            KRGen(BitSeq::from([0,1,0]), v, x(0)) =>  R::one(),
            KRGen(BitSeq::from([0,0,1]), v, x(0)) => -R::one(),
            KRGen(BitSeq::from([0,0,1]), v, x(1)) =>  R::one()
        });

        let h = BitSeq::from([0,1,0]);
        let z = KRGen(h, v, one.clone());
        let ys = cube.d_x(&z);

        assert_eq!(ys, map! {
            KRGen(BitSeq::from([1,1,0]), v, x(1)) => -R::one(),
            KRGen(BitSeq::from([1,1,0]), v, x(2)) =>  R::one(),
            KRGen(BitSeq::from([0,1,1]), v, x(0)) =>  R::one(),
            KRGen(BitSeq::from([0,1,1]), v, x(1)) => -R::one()
        });

        let h = BitSeq::from([1,1,1]);
        let z = KRGen(h, v, one.clone());
        let ys = cube.d_x(&z);

        assert!(ys.is_zero());
    }

    #[test]
    fn vert_gens() {
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,0]);
        let q = 0;

        let cube = make_cube(&l, v, q);
        let h = BitSeq::from([0, 0, 0]);
        let gens = cube.vert_gens(h);

        assert_eq!(gens.len(), 1);
        assert!(gens.iter().all(|x| x.0 == h));

        let h = BitSeq::from([1, 0, 0]);
        let gens = cube.vert_gens(h);

        assert_eq!(gens.len(), 3);
        assert!(gens.iter().all(|x| x.0 == h));
    }

    #[test]
    fn gens() {
        let l = Link::trefoil();
        let v = BitSeq::from([0,0,0]);
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
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);
        let c = cube.into_complex();

        assert_eq!(c[0].rank(), 1);
        assert_eq!(c[1].rank(), 12);
        assert_eq!(c[2].rank(), 26);
        assert_eq!(c[3].rank(), 15);

        c.check_d_all();

        let v = BitSeq::from([1,0,0]);
        let q = 1;
        let cube = make_cube(&l.mirror(), v, q);
        let c = cube.into_complex();
        
        assert_eq!(c[0].rank(), 6);
        assert_eq!(c[1].rank(), 40);
        assert_eq!(c[2].rank(), 70);
        assert_eq!(c[3].rank(), 36);
        
        c.check_d_all();
    }

    #[test]
    fn homology() { 
        let l = Link::trefoil();
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        let cube = make_cube(&l, v, q);

        let c = cube.into_complex();
        let h = c.homology(false);

        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 1);
        assert_eq!(h[3].rank(), 1);

        let l = Link::trefoil().mirror();
        let v = BitSeq::from([1,0,0]);
        let q = 1;
        let cube = make_cube(&l, v, q);
        let c = cube.into_complex();
        let h = c.homology(false);
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }
}