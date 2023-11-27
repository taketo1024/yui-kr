use std::sync::{Arc, OnceLock};

use num_traits::One;
use yui::{EucRing, EucRingOps};
use yui::bitseq::BitSeq;

use super::base::{KRGen, KRPoly, KRMono, KRChain, KRPolyChain, sign_between, combine, decombine};
use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

pub struct KRTotCube<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    data: Arc<KRCubeData<R>>,
    q_slice: isize,
    verts: Vec<OnceLock<KRHorHomol<R>>> // serialized for fast access
} 

impl<R> KRTotCube<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Arc<KRCubeData<R>>, q_slice: isize) -> Self {
        let n = data.dim();
        let verts = BitSeq::generate(n).map(|_|
            OnceLock::new()
        ).collect();

        Self {
            data,
            q_slice,
            verts
        }
    }

    pub fn dim(&self) -> usize { 
        self.data.dim()
    }

    pub fn q_slice(&self) -> isize { 
        self.q_slice
    }

    pub fn vert(&self, v_coords: BitSeq) -> &KRHorHomol<R> {
        let i = v_coords.as_u64() as usize;
        self.verts[i].get_or_init(||
            KRHorHomol::new(self.data.clone(), v_coords, self.q_slice)
        )
    }

    pub fn edge_poly(&self, h_coords: BitSeq, i: usize) -> KRPoly<R> {
        self.data.ver_edge_poly(h_coords, i)
    }

    #[inline(never)] // for profilability
    pub fn d(&self, z: &KRChain<R>) -> KRChain<R> { 
        let z = combine(z.clone());
        let n = self.dim();

        let dz = z.iter().flat_map(|(v, f)| { 
            (0..n).filter(|&i|
                v.1[i].is_zero()
            ).map(move |i| {
                let w = KRGen(v.0, v.1.edit(|b| b.set_1(i)), KRMono::one());
                let e = R::from_sign( sign_between(v.1, w.1) );
                let g = self.edge_poly(v.0, i);
                let h = f * g * e;
                (w, h)
            })
        }).collect::<KRPolyChain<_>>();

        decombine(dz)
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