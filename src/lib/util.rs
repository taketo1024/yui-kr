use std::collections::HashMap;
use std::ops::RangeInclusive;

use itertools::Itertools;
use yui::poly::LPoly;

use crate::KRHomologyStr;
pub type QPoly = LPoly<'q', i32>;

pub fn make_qpoly_map(str: &KRHomologyStr) -> HashMap<(isize, isize), QPoly> { 
    let elements = str.into_iter().into_group_map_by(|(idx, _)|
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
    });
    elements.collect()
}

pub fn make_qpoly_table(str: &KRHomologyStr) -> String {
    let polys = make_qpoly_map(str);   
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

pub fn mirror(res: &KRHomologyStr) -> KRHomologyStr { 
    res.into_iter().map(|(idx, &v)| ((-idx.0, -idx.1, -idx.2), v)).collect()
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