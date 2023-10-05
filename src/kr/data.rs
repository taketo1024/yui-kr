use std::collections::HashSet;
use std::marker::PhantomData;
use petgraph::{Graph, algo::min_spanning_tree};

use yui_core::{Ring, RingOps};
use yui_link::{Link, LinkComp, CrossingType};

pub(crate) struct KRCubeData<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    dim: usize,
    x_signs: Vec<i32>,
    _base_ring: PhantomData<R>
}

impl<R> KRCubeData<R> 
where R: Ring, for<'x> &'x R: RingOps<R> {
    pub fn new(link: &Link) -> Self {
        let dim = link.crossing_num() as usize;
        let x_signs = link.crossing_signs();

        Self {
            dim,
            x_signs,
            _base_ring: PhantomData,
        }
    }

    pub fn dim(&self) -> usize { 
        self.dim
    }

    pub fn x_signs(&self) -> &Vec<i32> { 
        &self.x_signs
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

#[cfg(test)]
mod tests {
    use yui_ratio::Ratio;
    use yui_link::Link;
    use super::*;

    type R = Ratio<i64>;

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
        println!("{:?}", u.components());
    }
}