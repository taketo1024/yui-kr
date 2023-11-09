use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use cartesian::cartesian;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use yui_core::{EucRing, EucRingOps, isize2};
use yui_homology::ChainComplex2;
use yui_homology::utils::ChainReducer;
use yui_lin_comb::LinComb;
use yui_matrix::sparse::{SpMat, SpVec, MatType};
use yui_utils::bitseq::BitSeq;

use super::base::{VertGen, BasePoly, sign_between};
use super::data::KRCubeData;
use super::hor_homol::KRHorHomol;

pub type KRTotComplex<R> = ChainComplex2<R>;

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

    pub fn q_slice(&self) -> isize { 
        self.q_slice
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

    fn hor_homol_gens(&self, i: usize, j: usize) -> Vec<LinComb<VertGen, R>> {
        self.data.verts(j).into_iter().flat_map(|v| {
            self.hor_hml(v).homol_gens(i)
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

    fn edge_poly(&self, h_coords: BitSeq, i: usize) -> BasePoly<R> {
        self.data.ver_edge_poly(h_coords, i)
    }

    fn d(&self, e: &VertGen) -> LinComb<VertGen, R> { 
        let (h0, v0) = (e.0, e.1);
        let x0 = &e.2;
        let n = self.data.dim();

        (0..n).filter(|&i| 
            v0[i].is_zero()
        ).flat_map(|i| {
            let v1 = v0.edit(|b| b.set_1(i));
            let e = R::from_sign( sign_between(v0, v1) );
            let p = BasePoly::from( (x0.clone(), e ) );
            let f = self.edge_poly(h0, i);
            let q = f * p;

            q.into_iter().map(move |(x1, r)| 
                (VertGen(h0, v1, x1), r)
            )
         }).collect()
    }

    fn d_matrix(&self, i: usize, j: usize) -> SpMat<R> { 
        let n = self.rank(i, j);
        let q = SpMat::id(n);
        Self::d_matrix_for(&self, i, j, &q)
    }

    fn d_matrix_for(&self, i: usize, j: usize, q: &SpMat<R>) -> SpMat<R> { 
        let gens = self.hor_homol_gens(i, j);
        let (m, n) = (self.rank(i, j+1), q.cols());

        let entries: Vec<_> = (0..n).into_par_iter().map(|l| { 
            let v = q.col_vec(l);
            let z = v.iter().map(|(k, a)| 
                &gens[k] * a
            ).sum::<LinComb<_, _>>();
            
            let dz = z.apply(|x| self.d(x));
            let w = self.vectorize(i, j + 1, &dz);

            w.iter().map(|(k, b)| 
                (k, l, b.clone())
            ).collect_vec()
        }).flatten().collect();

        SpMat::from_entries((m, n), entries)
    }

    pub fn into_complex(self, reducing: bool) -> KRTotComplex<R> {
        let n = self.data.dim() as isize;
        let range = cartesian!(0..=n, 0..=n).map(|(i, j)| isize2(i, j));
        let d_deg = isize2(0, 1);

        let mut reducer = ChainReducer::new(range, d_deg, true);

        for idx in reducer.support() {
            let (i, j) = (idx.0 as usize, idx.1 as usize);
            let d = if let Some(t) = reducer.trans(idx) {
                self.d_matrix_for(i, j, t.backward_mat())
            } else { 
                self.d_matrix(i, j)
            };
            reducer.set_matrix(idx, d);

            if reducing {
                reducer.reduce_at(idx);
            }
        }

        reducer.into_complex()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;
    use yui_homology::{ChainComplexTrait, RModStr};
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = BasePoly<R>;

    fn make_cube(l: &Link, q: isize) -> KRTotCube<R> {
        let data = KRCubeData::<R>::new(&l, 2);
        let rc = Arc::new(data);
        let cube = KRTotCube::new(rc, q);
        cube
    }

    #[test]
    fn edge_poly() { 
        let x = (0..3).map(|i| P::variable(i)).collect_vec();

        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let cube = make_cube(&l, 0);

        // p: neg, h: 0, v: 0 -> 1
        let h  = BitSeq::from([0,0,0]);
        let p = cube.edge_poly(h, 0); // 1
        assert_eq!(p, P::one());

        // p: neg, h: 1, v: 0 -> 1
        let h  = BitSeq::from([1,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
        assert_eq!(p, x[0]);

        let l = l.mirror(); // trefoil
        let cube = make_cube(&l, 0);

        // p: pos, h: 0, v: 0 -> 1
        let h  = BitSeq::from([0,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
        assert_eq!(p, x[0]);

        // p: pos, h: 1, v: 0 -> 1
        let h  = BitSeq::from([1,0,0]);
        let p = cube.edge_poly(h, 0); // x_bc
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
        let zs = cube.hor_homol_gens(3, 3);

        assert_eq!(zs.len(), 4);

        let _0 = R::from(0);
        let _1 = R::from(1);

        assert_eq!(cube.vectorize(3, 3, &zs[0]), SpVec::from(vec![_1,_0,_0,_0]));
        assert_eq!(cube.vectorize(3, 3, &zs[1]), SpVec::from(vec![_0,_1,_0,_0]));
        assert_eq!(cube.vectorize(3, 3, &(&zs[0] - &zs[3])), SpVec::from(vec![_1,_0,_0,-_1]));
    }

    #[test]
    fn complex() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let cube = make_cube(&l, q);
        let c = cube.into_complex(false);

        assert_eq!(c[(0, 0)].rank(), 0);
        assert_eq!(c[(0, 1)].rank(), 0);
        assert_eq!(c[(0, 2)].rank(), 0);
        assert_eq!(c[(0, 3)].rank(), 0);
        assert_eq!(c[(1, 0)].rank(), 0);
        assert_eq!(c[(1, 1)].rank(), 0);
        assert_eq!(c[(1, 2)].rank(), 0);
        assert_eq!(c[(1, 3)].rank(), 0);
        assert_eq!(c[(2, 0)].rank(), 0);
        assert_eq!(c[(2, 1)].rank(), 0);
        assert_eq!(c[(2, 2)].rank(), 0);
        assert_eq!(c[(2, 3)].rank(), 1);
        assert_eq!(c[(3, 0)].rank(), 0);
        assert_eq!(c[(3, 1)].rank(), 3);
        assert_eq!(c[(3, 2)].rank(), 6);
        assert_eq!(c[(3, 3)].rank(), 4);

        c.check_d_all();
    }

    #[test]
    fn complex_red() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let cube = make_cube(&l, q);
        let c = cube.into_complex(true);

        assert_eq!(c[(0, 0)].rank(), 0);
        assert_eq!(c[(0, 1)].rank(), 0);
        assert_eq!(c[(0, 2)].rank(), 0);
        assert_eq!(c[(0, 3)].rank(), 0);
        assert_eq!(c[(1, 0)].rank(), 0);
        assert_eq!(c[(1, 1)].rank(), 0);
        assert_eq!(c[(1, 2)].rank(), 0);
        assert_eq!(c[(1, 3)].rank(), 0);
        assert_eq!(c[(2, 0)].rank(), 0);
        assert_eq!(c[(2, 1)].rank(), 0);
        assert_eq!(c[(2, 2)].rank(), 0);
        assert_eq!(c[(2, 3)].rank(), 1);
        assert_eq!(c[(3, 0)].rank(), 0);
        assert_eq!(c[(3, 1)].rank(), 1);
        assert_eq!(c[(3, 2)].rank(), 0);
        assert_eq!(c[(3, 3)].rank(), 0);

        c.check_d_all();
    }

    #[test]
    fn homology() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let q = -4;
        let cube = make_cube(&l, q);
        let c = cube.into_complex(false);
        let h = c.homology(false);

        assert_eq!(h[(0, 0)].rank(), 0);
        assert_eq!(h[(0, 1)].rank(), 0);
        assert_eq!(h[(0, 2)].rank(), 0);
        assert_eq!(h[(0, 3)].rank(), 0);
        assert_eq!(h[(1, 0)].rank(), 0);
        assert_eq!(h[(1, 1)].rank(), 0);
        assert_eq!(h[(1, 2)].rank(), 0);
        assert_eq!(h[(1, 3)].rank(), 0);
        assert_eq!(h[(2, 0)].rank(), 0);
        assert_eq!(h[(2, 1)].rank(), 0);
        assert_eq!(h[(2, 2)].rank(), 0);
        assert_eq!(h[(2, 3)].rank(), 1);
        assert_eq!(h[(3, 0)].rank(), 0);
        assert_eq!(h[(3, 1)].rank(), 1);
        assert_eq!(h[(3, 2)].rank(), 0);
        assert_eq!(h[(3, 3)].rank(), 0);
    }
}