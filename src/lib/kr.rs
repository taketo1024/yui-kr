//! [`KRHomology`]: the reduced HOMFLY homology `ℋ(K)` of a knot, computed in
//! full and indexed by the triple grading `(i, j, k)`.

use std::ops::{Index, RangeInclusive};
use std::sync::Arc;

use delegate::delegate;
use yui_core::abst::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3, GenericSummand, Grid1};
use yui_link::{Link, Braid};

use crate::KRHomologyStr;
use crate::internal::data::KRCubeData;
use crate::internal::tot_homol::KRTotHomol;

/// The reduced HOMFLY homology of a knot, computed for every grading at once.
///
/// Indexing by `(i, j, k)` gives the summand `ℋ^{i,j,k}`, which is zero
/// outside [`support`](Self::support). Use [`KRCalc`](crate::KRCalc) instead
/// when the computation needs to be bounded or resumed.
pub struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    q_slices: Grid1<KRTotHomol<R>>,
    zero: GenericSummand<isize2, R>,
}

impl<R> KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    /// Computes `ℋ` of the closure of `b`. This is the preferred entry point,
    /// since the closure of a braid is what makes `ℋ(D)` an invariant.
    pub fn from_braid(b: &Braid) -> Self {
        let l = b.closure();
        Self::from_link(&l)
    }

    /// Computes `ℋ(D)` for the diagram `D` of a knot.
    ///
    /// note: H(D) is defined for any diagram, but is an invariant of the knot
    /// only when D is a braid closure [NS2024, Remark 2.2]. A minimal-crossing
    /// diagram may give a different answer -- 5_2 is such a case [NS2024, §5].
    pub fn from_link(link: &Link) -> Self {
        assert_eq!(link.n_comps(), 1, "only knots are supported.");
        let data = Arc::new(KRCubeData::new(link));
        Self::new(data)
    }

    /// Computes every `q`-slice of the cube `data`.
    pub fn new(data: Arc<KRCubeData<R>>) -> Self {
        let q_range = data.q_range();
        let q_slices = q_range.map(|q| {
            let range = data.hv_range(q);
            (q, KRTotHomol::new_restr(data.clone(), q, range))
        }).collect();
        let zero = GenericSummand::zero();
        Self { data, q_slices, zero }
    }

    // i, k: Morton bound [NS2024, Prop 2.6]; j: MFW inequality [Prop 2.3].
    // q indexes the slices of C(D) the computation actually runs over.
    delegate! {
        to self.data {
            pub fn i_range(&self) -> RangeInclusive<isize>;
            pub fn j_range(&self) -> RangeInclusive<isize>;
            pub fn k_range(&self) -> RangeInclusive<isize>;
            pub fn q_range(&self) -> RangeInclusive<isize>;
        }
    }

    /// The ranks at every grading where `ℋ` is non-zero.
    pub fn structure(&self) -> KRHomologyStr {
        self.support().filter_map(|&idx| {
            let r = self[idx].rank();

            if r > 0 {
                Some(((idx.0, idx.1, idx.2), r))
            } else {
                None
            }
        }).collect()
    }

    /// The `(j, k)`-table of Poincaré polynomials in `q`.
    pub fn to_table_string(&self) -> String {
        self.structure().qpoly_table()
    }
}

impl<R> KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    // The gradings not excluded a priori by the bounds above; `ℋ` vanishes
    // outside, but may also vanish at any of these.
    delegate! {
        to self.data {
            pub fn support(&self) -> std::collections::hash_set::Iter<'_, isize3>;
            pub fn is_supported(&self, idx: isize3) -> bool;
        }
    }
}

impl<R> Index<isize3> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = GenericSummand<isize2, R>;
    fn index(&self, idx: isize3) -> &Self::Output {
        match self.data.to_inner_grad(idx) {
            Some(isize3(q, h, v)) => &self.q_slices[q][isize2(h, v)],
            None => &self.zero
        }
    }
}

impl<R> Index<(isize, isize, isize)> for KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    type Output = GenericSummand<isize2, R>;
    fn index(&self, index: (isize, isize, isize)) -> &Self::Output {
        &self[isize3::from(index)]
    }
}

#[cfg(test)]
mod tests { 
    use maplit::hashmap;
    use yui_link::Braid;
    use yui_core::num::Ratio;
    use super::*;

    type R = Ratio<i64>;

    #[test]
    fn trivial() { 
        let b = Braid::from([1]);
        let h = KRHomology::<R>::from_braid(&b);

        assert_eq!(h[(0,0,0)].rank(), 1);
        assert_eq!(h[(1,0,0)].rank(), 0);

        let s = h.structure();

        assert_eq!(s.total_rank(), 1);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,0,0) => 1
        }));
    }

    #[test]
    fn b3_1() { 
        let b = Braid::from([1,1,1]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 3);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        }));
    }
    
    #[test]
    fn b4_1() { 
        let b = Braid::from([1,-2,1,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,-2,2) => 1,
            (-2,0,2) => 1,
            (0,2,-2) => 1,
            (0,0,0)  => 1,
            (2,0,-2) => 1
        }));
    }

    #[test]
    fn b5_1() { 
        let b = Braid::from([1,1,1,1,1]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 5);
        assert_eq!(s, KRHomologyStr::from(hashmap!{             
            (0,4,0)  => 1,
            (-2,6,0) => 1,
            (-4,4,4) => 1,
            (4,4,-4) => 1,
            (2,6,-4) => 1
        }));
    }

    #[test]
    fn b5_2() { 
        let b = Braid::from([1,1,1,2,-1,2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 7);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (2,4,-4) => 1,
            (2,2,-2) => 1,
            (0,4,-2) => 1,
            (-2,4,0) => 1,
            (0,2,0)  => 1,
            (0,6,-4) => 1,
            (-2,2,2) => 1
        }));
    }

    #[test]
    fn b6_1() { 
        let b = Braid::from([1,1,2,-1,-3,2,-3]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 9);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,-2,2) => 1,
            (0,0,0)  => 2,
            (2,0,-2) => 1,
            (2,2,-4) => 1,
            (0,4,-4) => 1,
            (0,2,-2) => 1,
            (-2,2,0) => 1,
            (-2,0,2) => 1    
        }));
    }

    #[test]
    fn b6_2() { 
        let b = Braid::from([1,1,1,-2,1,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 11);
        assert_eq!(s, KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (2,4,-4) => 1,
            (-2,4,0) => 1,
            (0,2,0)  => 2,
            (2,2,-2) => 1,
            (2,0,0)  => 1,
            (-2,2,2) => 1,
            (-2,0,4) => 1,
            (4,2,-4) => 1,
            (-4,2,4) => 1
        }));
    }

    #[test]
    fn b6_3() {
        let b = Braid::from([1,1,-2,1,-2,-2]);
        let h = KRHomology::<R>::from_braid(&b);
        let s = h.structure();

        assert_eq!(s.total_rank(), 13);
        assert_eq!(s, KRHomologyStr::from(hashmap!{
            (4,0,-4)  => 1,
            (2,-2,0)  => 1,
            (0,0,0)   => 3,
            (2,2,-4)  => 1,
            (-2,2,0)  => 1,
            (0,2,-2)  => 1,
            (0,-2,2)  => 1,
            (-2,0,2)  => 1,
            (-4,0,4)  => 1,
            (2,0,-2)  => 1,
            (-2,-2,4) => 1
        }));
    }

    // H^{i,j,k} ≅ H^{-i,j,k+2i} [NS2024, Prop 2.4].
    //
    // Only i ≤ 0 is ever computed -- `to_inner_grad` folds the rest -- so
    // asserting the symmetry on `structure()` would be circular. Instead
    // rebuild the i > 0 half from its own inner grading, which always lands
    // in a different (q, h, v) cell, and for some targets in a q-slice that
    // was never built at all.
    #[test]
    fn prop_2_4_independent() {
        let l = Braid::from([1,1,1]).closure();
        let data = Arc::new( KRCubeData::<R>::new(&l) );
        let h = KRHomology::new(data.clone());

        let isize3(i0, j0, k0) = data.root_grad();
        let folded = h.q_range();

        let targets = data.support()
            .filter(|idx| idx.0 > 0)
            .copied()
            .collect::<Vec<_>>();

        assert!(!targets.is_empty());

        let mut unbuilt = 0;

        for idx in targets {
            let isize3(i, j, k) = idx;

            // the unfolded inner grading, bypassing to_inner_grad.
            let q = ((i - i0) - (j - j0)) / 2;
            let (a, b) = ((j - j0) / 2, (k - k0) / 2);

            if !folded.contains(&q) {
                unbuilt += 1;
            }

            let direct = KRTotHomol::new_restr(data.clone(), q, (a..=a, b..=b));

            assert_eq!(direct[(a, b)].rank(), h[idx].rank(), "at {idx:?}");
        }

        // at least one slice lay outside the computed q-range, so the check
        // is not merely reading back what KRHomology already built.
        assert!(unbuilt > 0);
    }

    // H^{i,j,k}(m(K)) ≅ H^{-i,-j,-k}(K) [NS2024, Prop 2.5].
    // The mirror braid builds a different complex throughout -- opposite
    // signs, shifts, edge polys and exclusions -- so this cross-checks the
    // whole sign-convention stack.
    #[test]
    fn mirror_duality_3_1() {
        let s  = KRHomology::<R>::from_braid(&Braid::from([ 1, 1, 1])).structure();
        let ms = KRHomology::<R>::from_braid(&Braid::from([-1,-1,-1])).structure();

        assert_eq!(ms, s.mirror());
    }

    #[test]
    fn mirror_duality_5_2() {
        let s  = KRHomology::<R>::from_braid(&Braid::from([ 1, 1, 1, 2,-1, 2])).structure();
        let ms = KRHomology::<R>::from_braid(&Braid::from([-1,-1,-1,-2, 1,-2])).structure();

        assert_eq!(ms, s.mirror());
    }

    #[test]
    fn amphichiral_4_1() {
        let s  = KRHomology::<R>::from_braid(&Braid::from([ 1,-2, 1,-2])).structure();
        let ms = KRHomology::<R>::from_braid(&Braid::from([-1, 2,-1, 2])).structure();

        assert_eq!(ms, s.mirror()); // Prop 2.5
        assert_eq!(ms, s);          // 4_1 is amphichiral
    }

    #[test]
    fn index_out_of_support() {
        let h = KRHomology::<R>::from_braid(&Braid::from([1,1,1]));

        // gradings of the wrong parity have no inner index at all.
        for idx in [(1,0,0), (1,1,1), (0,1,0), (3,0,0), (100,100,100)] {
            let idx = isize3::from(idx);
            assert!(!h.is_supported(idx));
            assert_eq!(h[idx].rank(), 0, "at {idx:?}");
        }

        assert_eq!(h[(0,4,-2)].rank(), 1);
    }
}