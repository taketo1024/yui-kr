//! [`KRHomologyStr`]: the computed ranks of `ℋ`, and the invariants read off
//! them -- the HOMFLY polynomial, the Poincaré polynomial and the Δ-slices.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use itertools::Itertools;
use num_integer::Integer;
use serde::ser::SerializeSeq;
use yui_core::poly::{LPoly, LPoly2, LPoly3, Var3, Var2, Mono};

/// Laurent polynomials in `q`.
pub type QPoly = LPoly<'q', i32>;
/// Laurent polynomials in `q, a` — the HOMFLY polynomial lives here.
pub type QAPoly = LPoly2<'q', 'a', i32>;
/// Laurent polynomials in `t, q, a` — the Poincaré polynomial lives here.
pub type TQAPoly = LPoly3<'t', 'q', 'a', i32>;

/// `dim ℋ^{i,j,k}` at each grading, as far as it is known.
///
/// A grading maps to `Some(r)` once computed, and to `None` while it is still
/// non-determined — a partial run leaves the gradings it skipped in the latter
/// state, so they are not mistaken for zero. Gradings absent from the table
/// are known to be zero; see [`get`](Self::get).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KRHomologyStr(HashMap<(isize, isize, isize), Option<usize>>);

impl From<HashMap<(isize, isize, isize), usize>> for KRHomologyStr {
    fn from(value: HashMap<(isize, isize, isize), usize>) -> Self {
        Self(value.into_iter().map(|(k, v)| (k, Some(v))).collect())
    }
}

impl From<HashMap<(isize, isize, isize), Option<usize>>> for KRHomologyStr {
    fn from(value: HashMap<(isize, isize, isize), Option<usize>>) -> Self {
        Self(value.into_iter().collect())
    }
}

impl FromIterator<((isize, isize, isize), usize)> for KRHomologyStr {
    fn from_iter<T: IntoIterator<Item = ((isize, isize, isize), usize)>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<HashMap<_, _>>())
    }
}

impl FromIterator<((isize, isize, isize), Option<usize>)> for KRHomologyStr {
    fn from_iter<T: IntoIterator<Item = ((isize, isize, isize), Option<usize>)>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<HashMap<_, _>>())
    }
}

impl KRHomologyStr { 
    pub fn indices(&self) -> impl Iterator<Item = &(isize, isize, isize)> { 
        self.0.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(isize, isize, isize), &Option<usize>)> { 
        self.0.iter()
    }

    /// Whether every grading in the table has been computed.
    pub fn is_determined(&self) -> bool {
        self.0.values().all(Option::is_some)
    }

    pub fn iter_determined(&self) -> impl Iterator<Item = (&(isize, isize, isize), &usize)> { 
        self.0.iter().filter_map(|(idx, r)| r.as_ref().map(|r| (idx, r)))
    }

    pub fn non_determined(&self) -> impl Iterator<Item = &(isize, isize, isize)> { 
        self.0.iter().filter_map(|(idx, r)| 
            if r.is_some() { None } else { Some(idx) }
        )
    }

    /// `Some(r)` if the rank at `idx` is known, `None` if not yet computed.
    /// An absent grading is a known zero, so it gives `Some(0)`.
    pub fn get(&self, idx: (isize, isize, isize)) -> Option<usize> {
        // Some(Some(r)) -> Some(r) // determined
        // Some(None)    -> None    // non-determined
        // None          -> Some(0) // determined
        if let Some(e) = self.0.get(&idx) {
            e.to_owned()
        } else { 
            Some(0)
        }
    }

    /// Records the rank at `idx`, and at its Prop 2.4 reflection.
    /// A value of `0` removes both rather than storing them.
    pub fn set(&mut self, idx: (isize, isize, isize), value: usize) {
        // note: the reflected index holds the same value by the symmetry
        // H^{i,j,k} ≅ H^{-i,j,k+2i} [NS2024, Prop 2.4].
        let idx2 = (-idx.0, idx.1, idx.2 + 2 * idx.0);

        if value > 0 { 
            self.0.insert(idx,  Some(value));
            self.0.insert(idx2, Some(value));
        } else { 
            self.0.remove(&idx);
            self.0.remove(&idx2);
        }
    }

    /// `ℋ(m(K))` from `ℋ(K)`, by `(i, j, k) ↦ (-i, -j, -k)`
    /// [NS2024, Prop 2.5].
    pub fn mirror(&self) -> KRHomologyStr {
        self.iter().map(|(idx, &v)| ((-idx.0, -idx.1, -idx.2), v)).collect()
    }

    /// The total dimension over all determined gradings.
    pub fn total_rank(&self) -> usize {
        self.iter_determined().map(|(_, r)| r).sum()
    }

    /// The Δ-gradings `Δ = i + j + k` that carry generators, in order.
    pub fn delta(&self) -> Vec<isize> {
        self.indices().map(|(i, j, k)| i + j + k).unique().sorted().collect_vec()
    }

    /// Whether `ℋ` sits in a single Δ-grading, i.e. `K` is KR-thin.
    pub fn is_thin(&self) -> bool {
        self.delta().len() <= 1
    }

    /// `Σ t^{(k-j)/2} qⁱ aʲ dim ℋ^{i,j,k}`.
    pub fn poincare_poly(&self) -> TQAPoly {
        self.iter_determined().map(|(idx, &r)| { 
            let &(i, j, k) = idx;
            let x = Var3::from(((k - j)/2, i, j));
            let r = r as i32;
            (x, r)
        }).collect()
    }
    
    /// The HOMFLY polynomial `P(K)(q, a)`, as the graded Euler characteristic
    /// `Σ (-1)^{(k-j)/2} qⁱ aʲ dim ℋ^{i,j,k}` [NS2024, Thm 3].
    pub fn homfly_poly(&self) -> QAPoly {
        self.poincare_poly().into_iter().map(|(x, r)|{
            let (t, i, j) = x.deg();
            let x = Var2::from((i, j)); // q^i a^j
            let r = if t.is_even() { r } else { -r }; // t ↦ -1
            (x, r)
        }).collect::<QAPoly>()
    }
    
    /// A `(j, k)`-table whose entries are the Poincaré polynomials in `q`.
    pub fn qpoly_table(&self) -> String {
        let polys = self.iter_determined().into_group_map_by(|(idx, _)|
            (idx.1, idx.2) // (j, k)
        ).into_iter().map(|(jk, list)| { 
            let q = QPoly::mono;
            let elems = list.into_iter().map(|(idx, &a)| {
                let i = idx.0;
                let a = a as i32;
                (q(i), a) // a.q^i
            });
            let p = QPoly::from_iter(elems);
            (jk, p)
        }).collect::<HashMap<(isize, isize), QPoly>>();

        let j_range = range(polys.keys().map(|idx| idx.0)).step_by(2);
        let k_range = range(polys.keys().map(|idx| idx.1)).rev().step_by(2);
    
        yui_core::util::format::table("k\\j", k_range, j_range, |&k, &j| { 
            if let Some(p) = polys.get(&(j, k)) { 
                p.to_string()
            } else { 
                ".".to_string()
            }
        })
    }

    /// One `(i, j)`-table per Δ-grading, with `k = Δ - i - j` and `?` for the
    /// gradings still non-determined. This is the form used in [NS2024, §4].
    pub fn delta_table(&self) -> String {
        range(self.delta()).step_by(2).map(|d| { 
            let indices = self.indices().filter(|(i, j, k)| i + j + k == d).collect_vec();
            let i_range = range(indices.iter().map(|idx| idx.0)).step_by(2);
            let j_range = range(indices.iter().map(|idx| idx.1)).rev().step_by(2);
        
            format!("Δ: {d}\n") + 
            &yui_core::util::format::table("j\\i", j_range, i_range, |&j, &i| { 
                let k = d - (i + j);
                if let Some(p) = self.0.get(&(i, j, k)) { 
                    if let Some(r) = p { r.to_string() } else { "?".to_string() }
                } else { 
                    ".".to_string()
                }
            })
        }).join("\n")
    }
}

impl serde::Serialize for KRHomologyStr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut seq = serializer.serialize_seq(None)?;
        for ((i, j, k), r) in self.iter() { 
            let e = (i, j, k, r);
            seq.serialize_element(&e)?;
        }
        seq.end()
    }
}

impl<'de> serde::Deserialize<'de> for KRHomologyStr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let seq: Vec<(isize, isize, isize, Option<usize>)> = Vec::deserialize(deserializer)?;
        let str = seq.into_iter().map(|(i, j, k, r)| ((i, j, k), r)).collect();
        Ok(str)
    }
}

fn range<Itr>(itr: Itr) -> RangeInclusive<isize>
where Itr: IntoIterator<Item = isize> {
    if let Some((l, r)) = itr.into_iter().fold(None, |res, i| { 
        if let Some((mut l, mut r)) = res { 
            if i < l { l = i }
            if r < i { r = i }
            Some((l, r))
        } else { 
            Some((i, i))
        }
    }) { 
        l..=r
    } else { 
        0..=0
    }
}

#[cfg(test)]
mod tests {
    use maplit::hashmap;
    use super::*;
 
    #[test]
    fn non_determined() { 
        let s = KRHomologyStr::from(hashmap!{             
            (0,4,0)  => Some(1),
            (-2,6,0) => Some(1),
            (-4,4,4) => Some(1),
            (4,4,-4) => Some(1),
            (2,6,-4) => None
        });

        assert!(!s.is_determined());
        assert_eq!(s.total_rank(), 4);
        assert_eq!(s.non_determined().collect_vec(), vec![&(2,6,-4)]);
    }

    // H(3_1) = Q{-2,2,2} + Q{0,4,-2} + Q{2,2,-2}  [NS2024, §4]
    fn trefoil() -> KRHomologyStr {
        KRHomologyStr::from(hashmap!{
            (0, 4,-2) => 1_usize,
            (-2,2, 2) => 1,
            (2, 2,-2) => 1
        })
    }

    #[test]
    fn set_maintains_pairing() {
        let mut s = KRHomologyStr::from(HashMap::<(isize, isize, isize), usize>::new());

        // set also writes the Prop 2.4 reflection (-i, j, k + 2i).
        s.set((2, 2, -2), 1);
        assert_eq!(s.get((2, 2, -2)), Some(1));
        assert_eq!(s.get((-2, 2, 2)), Some(1));

        // i = 0 is its own reflection.
        s.set((0, 4, -2), 3);
        assert_eq!(s.get((0, 4, -2)), Some(3));
        assert_eq!(s.total_rank(), 5);

        // setting 0 clears both.
        s.set((2, 2, -2), 0);
        assert_eq!(s.get((2, 2, -2)), Some(0));
        assert_eq!(s.get((-2, 2, 2)), Some(0));
        assert_eq!(s.total_rank(), 3);
    }

    #[test]
    fn get_semantics() {
        let s = KRHomologyStr::from(hashmap!{
            (0, 0, 0) => Some(2),
            (2, 0, 0) => None
        });

        assert_eq!(s.get((0, 0, 0)), Some(2)); // determined
        assert_eq!(s.get((2, 0, 0)), None);    // non-determined
        assert_eq!(s.get((4, 4, 4)), Some(0)); // absent means zero
        assert!(!s.is_determined());
    }

    #[test]
    fn mirror_involution() {
        let s = trefoil();
        let m = s.mirror();

        // (i, j, k) ↦ (-i, -j, -k)  [NS2024, Prop 2.5]
        assert_eq!(m.get((0, -4, 2)), Some(1));
        assert_eq!(m.get((2, -2, -2)), Some(1));
        assert_eq!(m.get((-2, -2, 2)), Some(1));

        assert_eq!(m.mirror(), s);
        assert_eq!(m.total_rank(), s.total_rank());
    }

    #[test]
    fn delta_and_thin() {
        let s = trefoil();

        // Δ = i + j + k is 2 on every generator, so 3_1 is KR-thin.
        assert_eq!(s.delta(), vec![2]);
        assert!(s.is_thin());

        let mut t = s.clone();
        t.set((0, 0, 0), 1);

        assert_eq!(t.delta(), vec![0, 2]);
        assert!(!t.is_thin());
    }

    #[test]
    fn homfly_poly_trefoil() {
        let m = QAPoly::mono;

        // P(q, a) = Σ (-1)^{(k-j)/2} q^i a^j  [NS2024, Thm 3]
        //         = q^{-2} a^2 + q^2 a^2 - a^4,
        // which is the HOMFLY polynomial of the right-handed trefoil,
        // 2a^2 - a^4 + a^2 z^2 with z = q - q^{-1}.
        let expect = QAPoly::from_iter([
            (m(-2, 2),  1),
            (m( 2, 2),  1),
            (m( 0, 4), -1)
        ]);

        assert_eq!(trefoil().homfly_poly(), expect);
    }

    #[test]
    fn serialize_non_determined() {
        let s = KRHomologyStr::from(hashmap!{             
            (0,4,0)  => Some(1),
            (-2,6,0) => Some(1),
            (-4,4,4) => Some(1),
            (4,4,-4) => Some(1),
            (2,6,-4) => None
        });

        let ser = serde_json::to_string(&s).unwrap();
        let des: KRHomologyStr = serde_json::from_str(&ser).unwrap();
        assert_eq!(s, des);
    }
}