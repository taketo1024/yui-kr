#[path = "../src/bin/app/mod.rs"]
pub mod app;

use log::info;
use yui_kr::util::QAPoly;

#[test]
fn check_homfly_10() -> Result<(), Box<dyn std::error::Error>> { 
    check_homfly("homfly-10")
}

#[test]
fn check_homfly_11() -> Result<(), Box<dyn std::error::Error>> { 
    check_homfly("homfly-11")
}

fn check_homfly(target: &str) -> Result<(), Box<dyn std::error::Error>>  {
    use app::utils::*;

    let proj_dir = std::env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{proj_dir}/data/");
    let mut reader = csv::Reader::from_path(&format!("{data_dir}/{target}.csv"))?;

    for r in reader.records() { 
        let r = r?;
        let name = &r[0];

        if !result_exists(name) { 
            info!("skip: {name}");
            continue
        }

        let data = &r[1];
        let data = data.replace(";", ",");
        let answer = parse_homfly(&data)?;

        let result = load_result(name)?;
        let poly = yui_kr::util::homfly_poly(&result);

        assert_eq!(answer, poly);
    }

    Ok(())
}

// KnotInfo, Polynomial vector notation
// see: https://knotinfo.math.indiana.edu/descriptions/homfly_polynomial_vector.html
//
//  {0, 1, {1, 2, 2, -1}, {1, 1, 1}} 
//   = (2*v^2 + -1*v^4) * z^0 + (1*v^2) * z^2
//
//  with v ↦ a, z ↦ q - q^{-1}.

pub fn parse_homfly(s: &str) -> Result<QAPoly, Box<dyn std::error::Error>> {
    use regex::Regex;
    use std::str::FromStr;
    use num_traits::Pow;
    use yui::AddMon;

    type P = QAPoly;

    let m = P::mono;
    let z = P::from_iter([(m(1, 0), 1), (m(-1, 0), -1)]); // q - q^{-1}
        
    let p1 = r"\{([-0-9]+), ([-0-9]+), (\{.*?\})\}";
    let p2 = r"\{([-0-9]+), ([-0-9]+), ([-0-9, ]+)*\}";
    let p3 = r", ";

    let r1 = Regex::new(p1).unwrap();
    let r2 = Regex::new(p2).unwrap();
    let r3 = Regex::new(p3).unwrap();

    let p = P::sum(r1.captures_iter(s).map(|c1| { 
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
                // dbg!(c);
                let v = m(0, e * 2);
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
    }));

    Ok(p)
}