use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use itertools::{Itertools, izip};
use petgraph::{Graph, algo::min_spanning_tree};

use yui_core::{Ring, RingOps, Sign, isize3, PowMod2, GetSign};
use yui_link::{Link, LinkComp, CrossingType, Crossing, Edge};
use yui_utils::bitseq::{BitSeq, Bit};

use super::base::BasePoly;

pub(crate) struct KRCubeData<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    n_cross: usize,
    writhe: isize, 
    n_seif: isize,
    x_signs: Vec<Sign>,
    x_polys: Vec<KRCubeX<R>>,
    _base_ring: PhantomData<R>
}

impl<R> KRCubeData<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(link: &Link) -> Self {
        let n = link.crossing_num() as usize;
        let w = link.writhe() as isize;
        let s = link.seifert_circles().len() as isize;
        let x_signs = link.crossing_signs();
        let x_polys = Self::collect_x_polys(link);

        Self {
            n_cross: n,
            writhe: w,
            n_seif: s,
            x_signs,
            x_polys,
            _base_ring: PhantomData,
        }
    }

    pub fn dim(&self) -> usize { 
        self.n_cross
    }

    pub fn x_signs(&self) -> &Vec<Sign> { 
        &self.x_signs
    }

    pub fn x_sign(&self, i: usize) -> Sign { 
        self.x_signs[i]
    }

    pub fn x_poly(&self, i: usize) -> &KRCubeX<R> {
        &self.x_polys[i]
    }

    pub fn all_verts(&self) -> Vec<BitSeq> { 
        BitSeq::generate(self.n_cross)
    }

    // TODO cache
    pub fn verts(&self, k: usize) -> Vec<BitSeq> {
        self.all_verts().into_iter().filter(|v| v.weight() == k).collect()
    }

    // TODO cache
    pub fn targets(&self, from: BitSeq) -> Vec<BitSeq> { 
        let n = self.n_cross;
        (0..n).filter(|&i| from[i].is_zero() ).map(|i| { 
            let mut t = from.clone();
            t.set_1(i);
            t
        }).collect()
    }    

    // TODO cache
    pub fn edge_sign(&self, from: BitSeq, to: BitSeq) -> Sign { 
        sign_between(from, to)
    }

    fn l_grad_shift(&self) -> isize3 { 
        let w = self.writhe;
        let s = self.n_seif;
        isize3(-w + s - 1, w + s - 1, w - s + 1)
    }

    fn x_grad_shift(x_sign: Sign, h: Bit, v: Bit) -> isize3 { 
        use Bit::{Bit0, Bit1};
        if x_sign.is_positive() {
            match (h, v) {
                (Bit0, Bit0) => isize3(2, -2, -2),
                (Bit1, Bit0) => isize3(0,  0, -2),
                (Bit0, Bit1) => isize3(0, -2,  0),
                (Bit1, Bit1) => isize3(0,  0,  0)
            }
        } else { 
            match (h, v) {
                (Bit0, Bit0) => isize3( 0, -2, 0),
                (Bit1, Bit0) => isize3( 0,  0, 0),
                (Bit0, Bit1) => isize3( 0, -2, 2),
                (Bit1, Bit1) => isize3(-2,  0, 2)
            }
        }
    }

    pub fn root_grad(&self) -> isize3 { 
        let n = self.n_cross;
        let zero = BitSeq::zeros(n);
        self.grad_at(zero, zero)
    }

    pub fn grad_at(&self, h_coords: BitSeq, v_coords: BitSeq) -> isize3 { 
        let n = self.n_cross;

        assert_eq!(h_coords.len(), n);
        assert_eq!(v_coords.len(), n);

        let init = self.l_grad_shift();
        let trip = izip!(
            self.x_signs.iter(), 
            h_coords.iter(), 
            v_coords.iter()
        );

        trip.fold(init, |grad, (&e, h, v)| { 
            grad + Self::x_grad_shift(e, h, v)
        })
    }

    // Morton bound (only for knots)
    pub fn i_range(&self) -> RangeInclusive<isize> { 
        let n = self.n_cross as isize;
        let s = self.n_seif;
        -n + s - 1 ..= n - s + 1
    }

    // MFW inequality
    pub fn j_range(&self) -> RangeInclusive<isize> { 
        let w = self.writhe;
        let s = self.n_seif;
        w - s + 1 ..= w + s - 1
    }

    pub fn k_range(&self) -> RangeInclusive<isize> { 
        self.i_range()
    }

    pub fn is_triv(&self, grad: isize3) -> bool { 
        let isize3(i, j, k) = grad;
        let isize3(i0, j0, k0) = self.root_grad();

        let i_range = self.i_range();
        let j_range = self.j_range();
        let k_range = self.k_range();

        let ok = (i - i0).is_even() 
            && (j - j0).is_even() 
            && (k - k0).is_even()
            && i_range.contains(&i)
            && j_range.contains(&j)
            && k_range.contains(&k)
            && k_range.contains(&(k + 2 * i));

        !ok
    }

    pub fn to_inner_grad(&self, grad: isize3) -> Option<isize3> { 
        let isize3(i, j, k) = grad;
        let isize3(i0, j0, k0) = self.root_grad();
        
        if (i - i0).is_even() && (j - j0).is_even() && (k - k0).is_even() {
            let h = (j - j0) / 2;
            let v = (k - k0) / 2;
            let q = ((i - i0) - (j - j0)) / 2;
            Some(isize3(h, v, q))
        } else { 
            None
        }
    }

    fn collect_x_polys(link: &Link) -> Vec<KRCubeX<R>> {
        let n = link.crossing_num() as usize;
        let signs = link.crossing_signs();

        // TODO support links
        assert_eq!(link.components().len(), 1); 

        // is_vert: 
        //          
        //  a(3) b(2)   c(3) a(2)
        //    ↑\ /↑        \ /→  
        //      X           X    
        //     / \         / \→   
        //  c(0) d(1)   d(0) b(1)
        //
        //    true        false
       
        let is_vert = |x: &Crossing, sign: Sign| -> bool { 
            use CrossingType::*;
            match (x.ctype(), sign.is_positive()) {
                (X, false) | (Xm, true)  | (V, _) => true,
                (X, true)  | (Xm, false) | (H, _) => false
            }
        };

        // collect edges and monomials.
        
        let (path, mons) = { 
            let mut path = vec![];
            let mut mons = vec![];

            link.traverse_edges((0, 0), |i, j| { 
                let x = &link.data()[i];
                let e = x.edge(j);
                let x_i = BasePoly::variable(i);
                let p_i = match (is_vert(x, signs[i]), j) {
                    (true, 0) | (false, 3) =>  x_i,
                    (true, 1) | (false, 0) => -x_i,
                    _ => panic!()
                };

                path.push(e);
                mons.push(p_i);
            });

            assert_eq!(path.len(), 2 * n + 1);
            assert_eq!(path.first(), path.last());

            path.pop();
            mons.pop();

            (path, mons)
        };

        let traverse = |from: Edge, to: Edge| -> BasePoly<R> { 
            let m = path.len(); // == 2 * n
            let i = path.iter().find_position(|&&e| e == from).unwrap().0;
            let j = path.iter().find_position(|&&e| e == to  ).unwrap().0;
            let l = if i < j { 
                j - i
            } else { 
                m + j - i
            };
            (i .. i + l).map(|k| &mons[k % m]).sum()
        };

        (0 .. n).map(|i| {
            let x = &link.data()[i];
            let (a,b,c) = if is_vert(x, signs[i]) { 
                (3,2,0)
            } else { 
                (2,1,3)
            };

            let x_ac = traverse(x.edge(c), x.edge(a));
            let x_bc = traverse(x.edge(c), x.edge(b));
    
            KRCubeX { x_ac, x_bc }
        }).collect()
    }

    fn seifert_graph(link: &Link) -> Graph<LinkComp, usize> { 
        type G = Graph<LinkComp, usize>;
        let s0 = link.ori_pres_state();
        let l0 = link.resolved_by(&s0);
        let mut graph = Graph::new();

        for c in l0.components() {
            graph.add_node(c);
        }

        let find_node = |graph: &G, e| { 
            graph.node_indices().find(|&i| 
                graph[i].contains(e)
            )
        };

        for (i, x) in l0.data().iter().enumerate() {
            let (e1, e2) = if x.ctype() == CrossingType::V {
                (x.edge(0), x.edge(1))
            } else { 
                (x.edge(0), x.edge(2))
            };
            let n1 = find_node(&graph, e1).unwrap();
            let n2 = find_node(&graph, e2).unwrap();
            graph.add_edge(n1, n2, i);
        }

        graph
    }

    fn resolve(link: &Link) -> Link {
        let g = Self::seifert_graph(&link);
        let t = min_spanning_tree(&g).filter_map(|e| 
            match e { 
                petgraph::data::Element::Edge { 
                    source: _, 
                    target: _, 
                    weight: x 
                } => Some(x),
                _ => None
            }
        ).collect::<HashSet<_>>();

        let n = link.crossing_num() as usize;
        let s0 = link.ori_pres_state();
        let mut data = link.data().clone();

        for i in 0..n { 
            if t.contains(&i) { 
                continue
            }
            data[i].resolve(s0[i])
        }

        Link::new(data)
    }
}

/*
 *    a   b
 *     \ /
 *      X
 *     / \
 *    c   d
 * 
 * x_ac = x_a - x_c = -(x_b - x_d),
 * x_bc = x_b - x_c = -(x_a - x_d)
 */
pub(crate) struct KRCubeX<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub x_ac: BasePoly<R>,
    pub x_bc: BasePoly<R>
}

pub fn sign_between(from: BitSeq, to: BitSeq) -> Sign { 
    assert_eq!(from.len(), to.len());
    assert_eq!(to.weight() - from.weight(), 1);
    
    let n = from.len();
    let i = (0..n).find(|&i| from[i] != to[i]).unwrap();
    let e = (0..i).filter(|&j| from[j].is_one()).count() as i32;

    (-1).pow_mod2(e).sign()
}

#[cfg(test)]
mod tests {
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = BasePoly<R>;

    #[test]
    fn collect_x_polys() { 
        let l = Link::from_pd_code([[1,4,2,5],[5,2,6,3],[3,6,4,1]]); // trefoil
        let g = KRCubeData::<R>::collect_x_polys(&l);
        let x = (0..3).map(|i| P::variable(i)).collect_vec();

        assert_eq!(g.len(), 3);
        assert_eq!(g[0].x_bc, x[0]);
        assert_eq!(g[1].x_bc, x[1]);
        assert_eq!(g[2].x_bc, x[2]);
        assert_eq!(g[0].x_ac, -&x[1] + &x[2]);
        assert_eq!(g[1].x_ac, -&x[2] + &x[0]);
        assert_eq!(g[2].x_ac, -&x[0] + &x[1]);
    }

    #[test]
    fn seif_graph() { 
        let l = Link::trefoil();
        let g = KRCubeData::<R>::seifert_graph(&l);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn resolve() { 
        let l = Link::hopf_link();
        let u = KRCubeData::<R>::resolve(&l);
        assert_eq!(u.components().len(), 1);
    }

    #[test]
    fn targets() {
        let l = Link::trefoil();
        let data = KRCubeData::<R>::new(&l);

        let ts = data.targets(BitSeq::from_iter([0,0,0]));
        assert_eq!(ts, vec![
            BitSeq::from_iter([1,0,0]), 
            BitSeq::from_iter([0,1,0]), 
            BitSeq::from_iter([0,0,1])]
        );

        let ts = data.targets(BitSeq::from_iter([1,0,0]));
        assert_eq!(ts, vec![
            BitSeq::from_iter([1,1,0]), 
            BitSeq::from_iter([1,0,1])]
        );

        let ts = data.targets(BitSeq::from_iter([1,1,0]));
        assert_eq!(ts, vec![
            BitSeq::from_iter([1,1,1])]
        )
    }

    #[test]
    fn edge_sign() {
        use Sign::*;
        let l = Link::trefoil();
        let data = KRCubeData::<R>::new(&l);

        let e = data.edge_sign(
            BitSeq::from_iter([0,0,0]), 
            BitSeq::from_iter([1,0,0]));
        assert_eq!(e, Pos);

        let e = data.edge_sign(
            BitSeq::from_iter([0,0,0]), 
            BitSeq::from_iter([0,1,0]));
        assert_eq!(e, Pos);

        let e = data.edge_sign(
            BitSeq::from_iter([1,0,0]), 
            BitSeq::from_iter([1,1,0]));
        assert_eq!(e, Neg);

        let e = data.edge_sign(
            BitSeq::from_iter([0,1,0]), 
            BitSeq::from_iter([1,1,0]));
        assert_eq!(e, Pos);
    }

    #[test]
    fn grad_shift() { 
        let l = Link::trefoil(); // (w, s) = (-3, 2)
        let data = KRCubeData::<R>::new(&l);
        assert_eq!(data.l_grad_shift(), isize3(4,-2,-4));
    }

    #[test]
    fn grad_at() { 
        let l = Link::trefoil();
        let data = KRCubeData::<R>::new(&l);
        let grad0 = data.root_grad();
        assert_eq!(grad0, isize3(4,-8,-4));

        let h = BitSeq::from_iter([1,0,0]);
        let v = BitSeq::from_iter([0,1,0]);
        let grad1 = data.grad_at(h, v);
        assert_eq!(grad1, isize3(4,-6,-2));
    }
}

trait IsEven { 
    fn is_even(&self) -> bool;
}

impl IsEven for isize {
    fn is_even(&self) -> bool {
        self & 1 == 0
    }
}