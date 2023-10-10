use std::collections::HashSet;
use std::marker::PhantomData;
use itertools::Itertools;
use petgraph::{Graph, algo::min_spanning_tree};

use yui_core::{Ring, RingOps, PowMod2};
use yui_link::{Link, LinkComp, CrossingType, Crossing, Edge};
use yui_utils::bitseq::BitSeq;
use super::base::EdgeRing;

pub(crate) struct KRCubeData<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    dim: usize,
    x_signs: Vec<i32>,
    x_polys: Vec<KRCubeX<R>>,
    _base_ring: PhantomData<R>
}

impl<R> KRCubeData<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(link: &Link) -> Self {
        let dim = link.crossing_num() as usize;
        let x_signs = link.crossing_signs();
        let x_polys = Self::collect_x_polys(link);

        Self {
            dim,
            x_signs,
            x_polys,
            _base_ring: PhantomData,
        }
    }

    pub fn dim(&self) -> usize { 
        self.dim
    }

    pub fn x_signs(&self) -> &Vec<i32> { 
        &self.x_signs
    }

    pub fn x_sign(&self, i: usize) -> i32 { 
        self.x_signs[i]
    }

    pub fn x_poly(&self, i: usize) -> &KRCubeX<R> {
        &self.x_polys[i]
    }

    // TODO cache
    pub fn verts(&self, k: usize) -> Vec<BitSeq> {
        let n = self.dim;
        BitSeq::generate(n).into_iter().filter(|v| v.weight() == k).collect()
    }

    // TODO cache
    pub fn targets(&self, from: BitSeq) -> Vec<BitSeq> { 
        let n = self.dim;
        (0..n).filter(|&i| from[i].is_zero() ).map(|i| { 
            let mut t = from.clone();
            t.set_1(i);
            t
        }).collect()
    }    

    // TODO cache
    pub fn edge_sign(&self, from: BitSeq, to: BitSeq) -> i32 { 
        assert_eq!(to.weight() - from.weight(), 1);
        
        let n = self.dim;
        let i = (0..n).find(|&i| from[i] != to[i]).unwrap();
        let e = (0..i).filter(|&j| from[j].is_one()).count() as i32;

        (-1).pow_mod2(e)
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
       
        let is_vert = |x: &Crossing, sign: i32| -> bool { 
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
                let x_i = EdgeRing::variable(i);
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

        let traverse = |from: Edge, to: Edge| -> EdgeRing<R> { 
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
    pub x_ac: EdgeRing<R>,
    pub x_bc: EdgeRing<R>
}

#[cfg(test)]
mod tests {
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;
    type P = EdgeRing<R>;

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
        let l = Link::trefoil();
        let data = KRCubeData::<R>::new(&l);

        let e = data.edge_sign(
            BitSeq::from_iter([0,0,0]), 
            BitSeq::from_iter([1,0,0]));
        assert_eq!(e, 1);

        let e = data.edge_sign(
            BitSeq::from_iter([0,0,0]), 
            BitSeq::from_iter([0,1,0]));
        assert_eq!(e, 1);

        let e = data.edge_sign(
            BitSeq::from_iter([1,0,0]), 
            BitSeq::from_iter([1,1,0]));
        assert_eq!(e, -1);

        let e = data.edge_sign(
            BitSeq::from_iter([0,1,0]), 
            BitSeq::from_iter([1,1,0]));
        assert_eq!(e, 1);
    }
}