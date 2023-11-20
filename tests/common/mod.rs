#[allow(unused)]
#[path = "../../src/bin/app/mod.rs"]
pub mod app;

macro_rules! test {
    ($test:ident, $name:literal) => {
        #[test]
        #[ignore]
        fn $test() { 
            use common::app::*;
            let args = CliArgs { 
                target: $name.to_string(), 
                check_result: true,
                ..Default::default()
            };
            let app = App::new_with(args);
            app.run().unwrap();
        }
    };
}

pub(crate) use test;