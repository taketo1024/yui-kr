use log::{info, error};
use clap::Parser;
use super::utils::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct CliArgs {
    #[arg(long)]
    pub debug: bool
}

pub struct App {}

impl App { 
    pub fn new() -> Self { 
        App {}
    }

    pub fn run(&self) -> Result<String, i32> { 
        let args = CliArgs::parse();

        if args.debug { 
            self.init_logger();
        }

        info!("args: {:?}", args);

        let (res, time) = measure(||
            self.dispatch(&args)
        );

        let res = res.map_err(|e| { 
            error!("{}", e);
            eprintln!("\x1b[0;31merror\x1b[0m: {e}");
            1 // error code
        });

        info!("time: {:?}", time);

        res
    }

    fn init_logger(&self) {
        use simplelog::*;
        TermLogger::init(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto
        ).unwrap()
    }

    fn dispatch(&self, args: &CliArgs) -> Result<String, Box<dyn std::error::Error>> { 
        guard_panic(||
            Ok("todo".into())
        )
    }
}

#[derive(Debug, derive_more::Display)]
pub struct AppErr(pub(crate) String);

impl AppErr { 
    pub fn msg(&self) -> &str { 
        &self.0
    }
}

impl std::error::Error for AppErr {}

macro_rules! err {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        let e = AppErr(msg);
        std::result::Result::Err(e.into())
    }}
}

pub(crate) use err;