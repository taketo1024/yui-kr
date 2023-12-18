use std::collections::HashMap;
use std::ops::RangeInclusive;

use itertools::Itertools;
use num_integer::Integer;
use yui::poly::{LPoly, LPoly2, LPoly3, Var3, Var2, Mono};

pub type QPoly = LPoly<'q', i32>;
pub type QAPoly = LPoly2<'q', 'a', i32>;
pub type QATPoly = LPoly3<'q', 'a', 't', i32>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KRHomologyStr(HashMap<(isize, isize, isize), usize>);

impl From<HashMap<(isize, isize, isize), usize>> for KRHomologyStr {
    fn from(value: HashMap<(isize, isize, isize), usize>) -> Self {
        Self(value)
    }
}

impl FromIterator<((isize, isize, isize), usize)> for KRHomologyStr {
    fn from_iter<T: IntoIterator<Item = ((isize, isize, isize), usize)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for KRHomologyStr {
    type Item = ((isize, isize, isize), usize);
    type IntoIter = <HashMap<(isize, isize, isize), usize> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl KRHomologyStr { 
    pub fn inner(&self) -> &HashMap<(isize, isize, isize), usize> { 
        &self.0
    }

    pub fn total_rank(&self) -> usize { 
        self.0.values().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(isize, isize, isize), &usize)> { 
        self.0.iter()
    }

    pub fn mirror(&self) -> KRHomologyStr { 
        self.iter().map(|(idx, &v)| ((-idx.0, -idx.1, -idx.2), v)).collect()
    }
    
    pub fn poincare_poly(&self) -> QATPoly { 
        self.iter().map(|(idx, &r)| { 
            let &(i, j, k) = idx;
            let x = Var3::from((i, j, (k - j)/2));
            let r = r as i32;
            (x, r)
        }).collect()
    }
    
    pub fn homfly_poly(&self) -> QAPoly { 
        self.poincare_poly().iter().map(|(x, r)|{
            let (i, j, k) = x.deg();
            let x = Var2::from((i, j)); // q^i a^j
            let r = if k.is_even() { *r } else { -r }; // t ↦ -1
            (x, r)
        }).collect::<QAPoly>()
    }
    
    pub fn qpoly_map(&self) -> HashMap<(isize, isize), QPoly> { 
        self.iter().into_group_map_by(|(idx, _)|
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
        }).collect()
    }
    
    pub fn qpoly_table(&self) -> String {
        let polys = self.qpoly_map();   
        let j_range = range(polys.keys().map(|idx| idx.0)).step_by(2);
        let k_range = range(polys.keys().map(|idx| idx.1)).rev().step_by(2);
    
        yui::util::format::table("k\\j", k_range, j_range, |&k, &j| { 
            if let Some(p) = polys.get(&(j, k)) { 
                p.to_string()
            } else { 
                ".".to_string()
            }
        })
    }
}

fn range<Itr>(itr: Itr) -> RangeInclusive<isize>
where Itr: Iterator<Item = isize> {
    if let Some((l, r)) = itr.fold(None, |res, i| { 
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