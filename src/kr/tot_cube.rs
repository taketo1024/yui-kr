use std::cell::OnceCell;
use std::collections::HashMap;
use std::rc::Rc;

use num_traits::One;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{Idx2Iter, Idx2, GenericChainComplex};
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpMat, SpVec};
use yui_utils::bitseq::{BitSeq, Bit};

use super::base::{VertGen, EdgeRing};
use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

pub(crate) type KRTotComplex<R> = GenericChainComplex<R, Idx2Iter>;

pub(crate) struct KRTotCube<R>
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

    fn vectorize(&self, i: usize, j: usize, z: &LinComb<VertGen, R>) -> SpVec<R> {
        debug_assert!(z.iter().all(|(x, _)| 
            x.0.weight() == i && 
            x.1.weight() == j
        ));

        let (r, entries) = self.data.verts(j).iter().fold((0, vec![]), |acc, &v| { 
            let (mut r, mut entries) = acc;

            let z_v = z.filter_gens(|x| x.1 == v);
            let vec = self.hor_hml(v).vectorize(i, &z_v);
            for (i, a) in vec.iter() { 
                entries.push((i + r, a.clone()))
            }
            r += vec.dim();

            (r, entries)
        });

        SpVec::from_entries(r, entries)
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
        let mut entries = vec![];

        for (l, z) in gens.iter().enumerate() { 
            let dz: LinComb<VertGen, R> = z.iter().map(|(x, a)| { 
                self.differentiate(x) * a
            }).sum();
            let w = self.vectorize(i, j + 1, &dz);

            for (k, b) in w.iter() { 
                entries.push((k, l, b.clone()));
            }
        }

        SpMat::from_entries(shape, entries)
    }

    pub fn as_complex(self) -> KRTotComplex<R> {
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;
    use yui_homology::{HomologyComputable, RModStr};
    use yui_homology::test::ChainComplexValidation;
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

    fn make_cube(l: &Link, q: isize) -> KRTotCube<R> {
        let data = KRCubeData::<R>::new(&l);
        let rc = Rc::new(data);
        let cube = KRTotCube::new(rc, q);
        cube
    }

    #[test]
    fn edge_poly() { 
        let x = (0..3).map(|i| P::variable(i)).collect_vec();

        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let cube = make_cube(&l, 0);

        // p: neg, h: 0, v: 0 -> 1
        let h  = BitSeq::from_iter([0,0,0]);
        let v0 = BitSeq::from_iter([0,0,0]);
        let v1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h, v0, v1); // 1
        assert_eq!(p, P::one());

        // p: neg, h: 1, v: 0 -> 1
        let h  = BitSeq::from_iter([1,0,0]);
        let v0 = BitSeq::from_iter([0,0,0]);
        let v1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h, v0, v1); // x_bc
        assert_eq!(p, x[0]);

        let l = l.mirror(); // trefoil
        let cube = make_cube(&l, 0);

        // p: pos, h: 0, v: 0 -> 1
        let h  = BitSeq::from_iter([0,0,0]);
        let v0 = BitSeq::from_iter([0,0,0]);
        let v1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h, v0, v1); // x_bc
        assert_eq!(p, x[0]);

        // p: pos, h: 1, v: 0 -> 1
        let h  = BitSeq::from_iter([1,0,0]);
        let v0 = BitSeq::from_iter([0,0,0]);
        let v1 = BitSeq::from_iter([1,0,0]);
        let p = cube.edge_poly(h, v0, v1); // x_bc
        assert_eq!(p, P::one());
    }

    #[test]
    fn rank() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let cube = make_cube(&l, q);
        
        assert_eq!(cube.rank(0, 0), 0);
        assert_eq!(cube.rank(0, 1), 0);
        assert_eq!(cube.rank(0, 2), 0);
        assert_eq!(cube.rank(0, 3), 0);
        assert_eq!(cube.rank(1, 0), 0);
        assert_eq!(cube.rank(1, 1), 0);
        assert_eq!(cube.rank(1, 2), 0);
        assert_eq!(cube.rank(1, 3), 0);
        assert_eq!(cube.rank(2, 0), 1);
        assert_eq!(cube.rank(2, 1), 3);
        assert_eq!(cube.rank(2, 2), 6);
        assert_eq!(cube.rank(2, 3), 4);
        assert_eq!(cube.rank(3, 0), 1);
        assert_eq!(cube.rank(3, 1), 3);
        assert_eq!(cube.rank(3, 2), 6);
        assert_eq!(cube.rank(3, 3), 4);
    }

    #[test]
    fn vectorize() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = 0;
        let cube = make_cube(&l, q);
        let zs = cube.gens(3, 3);

        assert_eq!(zs.len(), 4);

        let _0 = R::from(0);
        let _1 = R::from(1);

        assert_eq!(cube.vectorize(3, 3, &zs[0]), SpVec::from(vec![_1,_0,_0,_0]));
        assert_eq!(cube.vectorize(3, 3, &zs[1]), SpVec::from(vec![_0,_1,_0,_0]));
        assert_eq!(cube.vectorize(3, 3, &(zs[0] - zs[3])), SpVec::from(vec![_1,_0,_0,-_1]));
    }

    #[test]
    fn complex() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let cube = make_cube(&l, q);
        let cpx = cube.as_complex();

        assert_eq!(cpx[Idx2(0, 0)].rank(), 0);
        assert_eq!(cpx[Idx2(0, 1)].rank(), 0);
        assert_eq!(cpx[Idx2(0, 2)].rank(), 0);
        assert_eq!(cpx[Idx2(0, 3)].rank(), 0);
        assert_eq!(cpx[Idx2(1, 0)].rank(), 0);
        assert_eq!(cpx[Idx2(1, 1)].rank(), 0);
        assert_eq!(cpx[Idx2(1, 2)].rank(), 0);
        assert_eq!(cpx[Idx2(1, 3)].rank(), 0);
        assert_eq!(cpx[Idx2(2, 0)].rank(), 0);
        assert_eq!(cpx[Idx2(2, 1)].rank(), 0);
        assert_eq!(cpx[Idx2(2, 2)].rank(), 0);
        assert_eq!(cpx[Idx2(2, 3)].rank(), 1);
        assert_eq!(cpx[Idx2(3, 0)].rank(), 0);
        assert_eq!(cpx[Idx2(3, 1)].rank(), 3);
        assert_eq!(cpx[Idx2(3, 2)].rank(), 6);
        assert_eq!(cpx[Idx2(3, 3)].rank(), 4);

        cpx.check_d_all();
    }

    #[test]
    fn homology() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let cube = make_cube(&l, q);
        let cpx = cube.as_complex();
        let hml = cpx.homology();

        assert_eq!(hml[Idx2(0, 0)].rank(), 0);
        assert_eq!(hml[Idx2(0, 1)].rank(), 0);
        assert_eq!(hml[Idx2(0, 2)].rank(), 0);
        assert_eq!(hml[Idx2(0, 3)].rank(), 0);
        assert_eq!(hml[Idx2(1, 0)].rank(), 0);
        assert_eq!(hml[Idx2(1, 1)].rank(), 0);
        assert_eq!(hml[Idx2(1, 2)].rank(), 0);
        assert_eq!(hml[Idx2(1, 3)].rank(), 0);
        assert_eq!(hml[Idx2(2, 0)].rank(), 0);
        assert_eq!(hml[Idx2(2, 1)].rank(), 0);
        assert_eq!(hml[Idx2(2, 2)].rank(), 0);
        assert_eq!(hml[Idx2(2, 3)].rank(), 1);
        assert_eq!(hml[Idx2(3, 0)].rank(), 0);
        assert_eq!(hml[Idx2(3, 1)].rank(), 1);
        assert_eq!(hml[Idx2(3, 2)].rank(), 0);
        assert_eq!(hml[Idx2(3, 3)].rank(), 0);
    }
}