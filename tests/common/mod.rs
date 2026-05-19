#![allow(unused)]

use std::path::PathBuf;
use yui_kr::KRHomologyStr;

#[path = "../../src/bin/app/mod.rs"]
pub mod app;

fn fixture_path(subdir: &str, name: &str, ext: &str) -> PathBuf {
    PathBuf::from(std::env!("CARGO_MANIFEST_DIR"))
        .join(subdir)
        .join(format!("{name}.{ext}"))
}

pub fn load_result(name: &str) -> Result<KRHomologyStr, Box<dyn std::error::Error>> {
    let path = fixture_path("results", name, "json");
    let str = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&str)?)
}

pub fn load_csv(name: &str) -> Result<csv::Reader<std::fs::File>, Box<dyn std::error::Error>> {
    let path = fixture_path("data", name, "csv");
    Ok(csv::Reader::from_path(path)?)
}

pub fn load_braid(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = load_csv("braid")?;
    for r in reader.records() {
        let r = r?;
        if &r[0] == name {
            return Ok(r[1].to_string());
        }
    }
    Err(format!("braid for {name} not found in data/braid.csv").into())
}

macro_rules! test {
    ($(#[$m:meta])* $test:ident, $name:literal) => {
        $(#[$m])*
        #[test]
        fn $test() -> Result<(), Box<dyn std::error::Error>> {
            use common::app::*;

            let answer = common::load_result($name)?;
            let braid = common::load_braid($name)?;

            let args = CliArgs {
                target: $name.to_string(),
                braid: Some(braid),
                int_type: IntType::BigInt,
                ..Default::default()
            };

            let app = App::new_with(args);
            let res = app.compute();

            match res { 
                Ok(res) => {
                    assert_eq!(res, answer);
                    Ok(())
                },
                Err(e) => {
                    Err(e)
                }
            }
        }
    };
}

pub(crate) use test;