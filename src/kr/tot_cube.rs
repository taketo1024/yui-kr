use std::cell::OnceCell;
use std::collections::HashMap;
use std::rc::Rc;

use num_traits::One;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{Idx2Iter, Idx2, GenericChainComplex};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::SpMat;
use yui_utils::bitseq::{BitSeq, Bit};

use super::base::{VertGen, EdgeRing};
use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

struct KRTotCube<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    data: Rc<KRCubeData<R>>,
    q_slice: isize,
    hor_hmls: HashMap<BitSeq, OnceCell<KRHorHomol<R>>>
} 

impl<R> KRTotCube<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn new(data: Rc<KRCubeData<R>>, q_slice: isize) -> Self {
        let n = data.dim();
        let hor_homols = BitSeq::generate(n).into_iter().map(|v|
            (v, OnceCell::new())
        ).collect();

        Self {
            data,
            q_slice,
            hor_hmls: hor_homols
        }
    }

    fn hor_hml(&self, v_coords: BitSeq) -> &KRHorHomol<R> {
        &self.hor_hmls[&v_coords].get_or_init(|| { 
            KRHorHomol::new(self.data.clone(), v_coords, self.q_slice)
        })
    }

    fn rank(&self, i: usize, j: usize) -> usize { 
        self.data.verts(j).into_iter().map(|v| {
            self.hor_hml(v).rank(i)
        }).sum()
    }

    fn gens(&self, i: usize, j: usize) -> Vec<&LinComb<VertGen, R>> {
        self.data.verts(j).into_iter().flat_map(|v| {
            self.hor_hml(v).gens(i)
        }).collect()
    }

    fn edge_poly(&self, h_coords: BitSeq, from: BitSeq, to: BitSeq) -> EdgeRing<R> {
        use Bit::{Bit0, Bit1};
        assert_eq!(to.weight() - from.weight(), 1);

        let n = from.len();
        let i = (0..n).find(|&i| from[i] != to[i]).unwrap();

        let sign = self.data.x_signs()[i];
        let h = h_coords[i];
        let p = self.data.x_poly(i);

        match (sign.is_positive(), h) {
            (true, Bit0) | (false, Bit1) => p.x_bc.clone(),
            (true, Bit1) | (false, Bit0) => EdgeRing::one()
        }
    }

    fn differentiate(&self, e: &VertGen) -> LinComb<VertGen, R> { 
        let VertGen(h, v, x) = e;
        let h = h.clone();
        let v0 = v.clone();
        
        self.data.targets(v0).into_iter().flat_map(|v1| { 
            let e = EdgeRing::from(self.data.edge_sign(v0, v1));
            let x = EdgeRing::from_term(x.clone(), R::one());
            let p = self.edge_poly(h, v0, v1);
            let q = e * x * p; // result polynomial

            q.into_iter().map(move |(x, r)| 
                (VertGen(h, v1, x), r)
            )
        }).collect()
    }

    fn d_matrix(&self, i: usize, j: usize) -> SpMat<R> { 
        let gens = self.gens(i, j);
        let shape = (self.rank(i, j+1), self.rank(i, j));

        SpMat::generate(shape, |_set| { 
            for (_l, z) in gens.iter().enumerate() { 
                let _dz: LinComb<VertGen, R> = z.iter().map(|(x, a)| { 
                    self.differentiate(x) * a
                }).sum();

                // vectorize dz and set entry.
                todo!()
            }
        })
    }

    pub fn as_complex(self) -> GenericChainComplex<R, Idx2Iter> {
        let n = self.data.dim() as isize;

        let start = Idx2(0, 0);
        let end   = Idx2(n, n);
        let range = start.iter_rect(end, (1, 1));
        
        GenericChainComplex::generate(range, Idx2(0, 1), |idx| {
            let (i, j) = idx.as_tuple();
            Some(self.d_matrix(i as usize, j as usize))
        })
    }
}