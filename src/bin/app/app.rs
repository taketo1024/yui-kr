use std::collections::HashMap;

use log::{info, error};
use clap::{Parser, ValueEnum};
use derive_more::Display;
use num_bigint::BigInt;
use yui::{Ratio, Integer, IntOps};
use yui_homology::DisplayTable;
use yui_kr::kr::KRHomology;
use yui_link::Braid;
use super::utils::*;

const MAX_BRAID_LEN: usize = 14;

#[derive(Parser, Debug, Default)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct CliArgs {
    pub target: String,

    #[arg(short, long, default_value = "i64")]
    pub int_type: IntType,

    #[arg(short, long)]
    pub mirror: bool,

    #[arg(short, long)]
    pub check_result: bool,

    #[arg(short, long)]
    pub save_result: bool,

    #[arg(long)]
    pub debug: bool
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
        Result::Err( Box::new(e) )
    }}
}

pub(crate) use err;

pub struct App {
    args: CliArgs
}

impl App { 
    pub fn new() -> Self { 
        let args = CliArgs::parse();
        Self::new_with(args)
    }

    pub fn new_with(args: CliArgs) -> Self { 
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

        if braid.len() >= MAX_BRAID_LEN { 
            err!("braid len {} of '{target}' exceeds limit: {MAX_BRAID_LEN}", braid.len())?;
        }

        let link = braid.closure();
        let link = if self.args.mirror { link.mirror() } else { link };

        let kr = KRHomology::<Ratio<I>>::new(&link);
        let res = kr.rank_all();

        if self.args.check_result { 
            Self::check_result(target, &res)?;
        }

        if self.args.save_result { 
            Self::save_data(target, &res)?;
        }

        let res = kr.display_table();

        Ok(res)
    }

    pub fn path_for(name: &str) -> String { 
        const RESULT_DIR: &str = "results/raw/";
        let dir = std::env!("CARGO_MANIFEST_DIR");
        format!("{dir}/{RESULT_DIR}{name}.json")
    }

    pub fn data_exists(name: &str) -> bool { 
        let path = Self::path_for(name);
        std::path::Path::new(&path).exists()
    }

    pub fn load_data(name: &str) -> Result<Res, Box<dyn std::error::Error>> {
        let path = Self::path_for(name);
        let json = std::fs::read_to_string(path)?;
        let list: Vec<((isize, isize, isize), usize)> = serde_json::from_str(&json)?;
        let data = list.into_iter().collect();
        Ok(data)
    }

    pub fn save_data(name: &str, data: &Res) -> Result<(), Box<dyn std::error::Error>> { 
        if Self::data_exists(name) { 
            info!("overwriting existing result: {}", Self::path_for(name));
        }
        let path = Self::path_for(name);
        let json = serde_json::to_string(&data.iter().collect::<Vec<_>>())?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn check_result(name: &str, data: &Res) -> Result<(), Box<dyn std::error::Error>> {
        if Self::data_exists(name) { 
            let expected = Self::load_data(name)?;
            if data != &expected { 
                err!("Incorrect result for {name}.\nComputed: {data:#?},\nExpected: {expected:#?}")?;
            }
        }
        Ok(())
    }
}

type Res = HashMap<(isize, isize, isize), usize>;