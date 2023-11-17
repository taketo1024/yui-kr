use log::{info, error};
use clap::{Parser, ValueEnum};
use derive_more::Display;
use num_bigint::BigInt;
use yui::{Ratio, Integer, IntOps};
use yui_homology::DisplayTable;
use yui_kr::kr::KRHomology;
use yui_link::Braid;
use super::utils::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct CliArgs {
    target: String,

    #[arg(short, long, default_value = "i64")]
    int_type: IntType,

    #[arg(long)]
    debug: bool
}

#[derive(ValueEnum, Clone, Copy, Default, Debug)]
#[clap(rename_all="lower")]
pub enum IntType { 
    #[default]
    I64, 
    I128, 
    BigInt
}

#[derive(Debug, Display)]
pub struct AppErr(pub(crate) String);
impl std::error::Error for AppErr {}

macro_rules! err {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        let e = AppErr(msg);
        std::result::Result::Err(e.into())
    }}
}

pub(crate) use err;

pub struct App {
    args: CliArgs
}

impl App { 
    pub fn new() -> Self { 
        let args = CliArgs::parse();
        if args.debug { 
            Self::init_logger();
        }
        App { args }
    }

    fn init_logger() {
        use simplelog::*;
        TermLogger::init(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto
        ).unwrap()
    }

    pub fn run(&self) -> Result<String, i32> { 
        info!("args: {:?}", self.args);

        let (res, time) = measure(|| guard_panic(||
            self.dispatch()
        ));

        let res = res.map_err(|e| { 
            error!("{}", e);
            eprintln!("\x1b[0;31merror\x1b[0m: {e}");
            1 // error code
        });

        info!("time: {:?}", time);

        res
    }

    fn dispatch(&self) -> Result<String, Box<dyn std::error::Error>> { 
        match self.args.int_type { 
            IntType::I64    => self.compute::<i64>(),
            IntType::I128   => self.compute::<i128>(),
            IntType::BigInt => self.compute::<BigInt>(),
        }
    }

    fn compute<I>(&self) -> Result<String, Box<dyn std::error::Error>>
    where I: Integer, for<'x> &'x I: IntOps<I> { 
        let target = &self.args.target;
        let braid = Braid::load(target)?;
        let link = braid.closure();

        let kr = KRHomology::<Ratio<I>>::new(&link);
        let res = kr.display_table();

        Ok(res)
    }
}