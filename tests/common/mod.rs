#[allow(unused)]
#[path = "../../src/bin/app/mod.rs"]
pub mod app;

macro_rules! test {
    ($(#[$m:meta])* $test:ident, $name:literal) => {
        $(#[$m])* 
        #[test]
        fn $test() { 
            use common::app::*;
            let args = CliArgs { 
                target: $name.to_string(), 
                int_type: IntType::BigInt,
                check_result: true,
                save_result: true,
                ..Default::default()
            };

            let app = App::new_with(args);
            let res = app.run();

            if let Some(e) = res.as_ref().err() { 
                eprintln!("error: {e}");
            }
            
            assert!(res.is_ok())
        }
    };
}

pub(crate) use test;