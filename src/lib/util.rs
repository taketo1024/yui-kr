use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::str::FromStr;
use itertools::Itertools;
use num_integer::Integer;
use num_traits::Pow;
use yui::AddMon;
use yui::poly::{LPoly, LPoly3, Var3, LPoly2, Mono, Var2};

use crate::KRHomologyStr;
pub type QPoly = LPoly<'q', i32>;
pub type QAPoly = LPoly2<'q', 'a', i32>;
pub type QATPoly = LPoly3<'q', 'a', 't', i32>;

pub fn poincare_poly(str: &KRHomologyStr) -> QATPoly { 
    str.iter().map(|(idx, &r)| { 
        let &(i, j, k) = idx;
        let x = Var3::from((i, j, (k - j)/2));
        let r = r as i32;
        (x, r)
    }).collect()
}

pub fn homfly_poly(str: &KRHomologyStr) -> QAPoly { 
    poincare_poly(str).iter().map(|(x, r)|{
        let (i, j, k) = x.deg();
        let x = Var2::from((i, j)); // q^i a^j
        let r = if k.is_even() { *r } else { -r }; // t ↦ -1
        (x, r)
    }).collect::<QAPoly>()
}

pub fn qpoly_map(str: &KRHomologyStr) -> HashMap<(isize, isize), QPoly> { 
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

pub fn qpoly_table(str: &KRHomologyStr) -> String {
    let polys = qpoly_map(str);   
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

// KnotInfo, Polynomial vector notation
// see: https://knotinfo.math.indiana.edu/descriptions/homfly_polynomial_vector.html
//
//  {0, 1, {1, 2, 2, -1}, {1, 1, 1}} 
//   = (2*v^2 + -1*v^4) * z^0 + (1*v^2) * z^2
//
//  with v ↦ a, z ↦ q - q^{-1}.
pub fn parse_homfly(s: &str) -> QAPoly {
    use regex::Regex;
    type P = QAPoly;

    let m = P::mono;
    let z = P::from_iter([(m(1, 0), 1), (m(-1, 0), -1)]); // q - q^{-1}
        
    let p1 = r"\{([-0-9]+), ([-0-9]+), (\{.*?\})\}";
    let p2 = r"\{([-0-9]+), ([-0-9]+), ([-0-9, ]+)*\}";
    let p3 = r", ";

    let r1 = Regex::new(p1).unwrap();
    let r2 = Regex::new(p2).unwrap();
    let r3 = Regex::new(p3).unwrap();

    P::sum(r1.captures_iter(s).map(|c1| { 
        // dbg!(&c1);
        let d0 = isize::from_str(&c1[1]).unwrap(); // (lowest degree of z) / 2.
        let d1 = isize::from_str(&c1[2]).unwrap(); // (highest degree of z) / 2.

        let v_part = &c1[3];
        let v_polys = r2.captures_iter(v_part).map(|c2| { 
            // dbg!(&c2);
            let e0 = isize::from_str(&c2[1]).unwrap(); // (lowest degree of v) / 2.
            let e1 = isize::from_str(&c2[2]).unwrap(); // (highest degree of v) / 2.
            let coeffs_part = &c2[3];

            debug_assert!(r3.split(coeffs_part).count() == (e1 - e0 + 1) as usize);

            let pairs = Iterator::zip(
                e0..=e1,
                r3.split(coeffs_part)
            );
            
            P::sum(pairs.map(|(e, c)| {
                let v = m(0, e * 2);
                dbg!(c);
                let c = i32::from_str(c).unwrap();
                P::from((v, c))
            }))
        });

        let pairs = Iterator::zip(
            d0..=d1,
            v_polys
        );

        P::sum(pairs.map(|(d, v)| { 
            assert!(d >= 0);
            v * z.pow(d * 2)
        }))
    }))
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

#[cfg(test)]
mod tests {
    use super::parse_homfly;
 
    #[test]
    fn test_parse_homfly() { 
        let s = "{0, 2, {2, 3, 3, -2}, {2, 3, 4, -1}, {2, 2, 1}}";
        let p = parse_homfly(&s);
        dbg!(&p);
    }
}