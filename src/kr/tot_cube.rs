use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use yui::{EucRing, EucRingOps};
use yui::bitseq::BitSeq;

use super::base::{KRGen, KRPoly, KRChain, sign_between};
use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

pub struct KRTotCube<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    data: Arc<KRCubeData<R>>,
    q_slice: isize,
    hor_hmls: HashMap<BitSeq, OnceLock<KRHorHomol<R>>>
} 

impl<R> KRTotCube<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self {
        let n = data.dim();
        let hor_homols = BitSeq::generate(n).into_iter().map(|v|
            (v, OnceLock::new())
        ).collect();

        Self {
            data,
            q_slice,
            hor_hmls: hor_homols
        }
    }

    pub fn dim(&self) -> usize { 
        self.data.dim()
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn vert(&self, v_coords: BitSeq) -> &KRHorHomol<R> {
        self.hor_hmls[&v_coords].get_or_init(|| { 
            KRHorHomol::new(self.data.clone(), v_coords, self.q_slice)
        })
    }

    pub fn edge_poly(&self, h_coords: BitSeq, i: usize) -> KRPoly<R> {
        self.data.ver_edge_poly(h_coords, i)
    }

    pub fn d(&self, e: &KRGen) -> KRChain<R> { 
        let (h0, v0) = (e.0, e.1);
        let x0 = &e.2;
        let n = self.data.dim();

        (0..n).filter(|&i| 
            v0[i].is_zero()
        ).flat_map(|i| {
            let v1 = v0.edit(|b| b.set_1(i));
            let e = R::from_sign( sign_between(v0, v1) );
            let p = KRPoly::from( (x0.clone(), e ) );
            let f = self.edge_poly(h0, i);
            let q = f * p;

            q.into_iter().map(move |(x1, r)| 
                (KRGen(h0, v1, x1), r)
            )
         }).collect()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use yui::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = KRPoly<R>;

    fn make_cube(l: &Link, q: isize) -> KRTotCube<R> {
        let data = Arc::new( KRCubeData::<R>::new(l, 2) );
        KRTotCube::new(data, q)
    }

    #[test]
    fn edge_poly() { 
        let x = P::variable;
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let cube = make_cube(&l, 0);

        // p: neg, h: 0, v: 0 -> 1
        let h  = BitSeq::from([0,0,0]);
        let p = cube.edge_poly(h, 0); // 1
        assert_eq!(p, P::one());

        // p: neg, h: 1, v: 0 -> 1
        let h  = BitSeq::from([1,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
        assert_eq!(p, x(0));

        let l = l.mirror(); // trefoil
        let cube = make_cube(&l, 0);

        // p: pos, h: 0, v: 0 -> 1
        let h  = BitSeq::from([0,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
        assert_eq!(p, x(0));

        // p: pos, h: 1, v: 0 -> 1
        let h  = BitSeq::from([1,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
        assert_eq!(p, P::one());
    }
}