#![allow(unused)]
#[path = "../../src/bin/app/mod.rs"]
pub mod app;

macro_rules! test {
    ($(#[$m:meta])* $test:ident, $name:literal) => {
        $(#[$m])* 
        #[test]
        fn $test() -> Result<(), Box<dyn std::error::Error>> { 
            use common::app::*;
            use common::app::utils::*;

            let answer = load_result($name)?;

            let args = CliArgs { 
                target: $name.to_string(), 
                int_type: IntType::BigInt,
                force_compute: true,
                ..Default::default()
            };

            let app = App::new_with(args);
            let res = app.compute();

            match res { 
                Ok(res) => {
                    assert_eq!(answer, res);
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