use log::{info, error};
use clap::{Parser, ValueEnum};
use derive_more::Display;
use num_bigint::BigInt;
use yui::{Ratio, EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr};
use yui_kr::kr::data::KRCubeData;
use yui_kr::{KRHomology, KRHomologyStr};
use yui_link::{Braid, Link};
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
    pub format: Output,

    #[arg(short = 'F', long)]
    pub force_compute: bool,

    #[arg(short, long)]
    pub check_result: bool,

    #[arg(short, long)]
    pub save_result: bool,

    #[arg(long)]
    pub no_multithread: bool,

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

#[derive(ValueEnum, Clone, Copy, Default, Debug)]
pub enum Output { 
    #[default]
    Table, 
    Poly,
    Homfly
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
        if args.no_multithread { 
            Self::set_multithread_enabled(false);
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

    fn set_multithread_enabled(flag: bool) { 
        yui_matrix::config::set_multithread_enabled(flag);
        yui_homology::config::set_multithread_enabled(flag);
        yui_kr::config::set_multithread_enabled(flag);
    }

    pub fn run(&self) -> Result<String, Box<dyn std::error::Error>> { 
        info!("args: {:?}", self.args);

        let n_threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
        info!("n-threads: {n_threads}");

        let (res, time) = measure(|| guard_panic(||
            self.compute()
        ));

        if let Some(e) = res.as_ref().err() { 
            error!("{e}");
        }

        let res = res?;
        let output = self.format(&res);

        info!("time: {:?}", time);

        Ok(output)
    }

    fn compute(&self) -> Result<KRHomologyStr, Box<dyn std::error::Error>> { 
        if !self.args.force_compute && result_exists(&self.args.target) { 
            info!("result exists: {}", self.args.target);
            let res = load_result(&self.args.target)?;
            let res = if self.args.mirror { res.mirror() } else { res };
            return Ok(res)
        }
        
        let link = self.make_link()?;

        let res = if link.writhe() > 0 { 
            info!("compute from mirror.");
            self.compute_kr_dispatch(&link.mirror()).mirror()
        } else { 
            self.compute_kr_dispatch(&link)
        };

        if self.args.check_result { 
            check_result(&self.args.target, &res)?;
        }
        
        if self.args.save_result { 
            save_result(&self.args.target, &res)?;
        }

        Ok(res)
    }

    fn compute_kr_dispatch(&self, link: &Link) -> KRHomologyStr { 
        match self.args.int_type { 
            IntType::I64    => self.compute_kr::<Ratio<i64>>(link),
            IntType::I128   => self.compute_kr::<Ratio<i128>>(link),
            IntType::BigInt => self.compute_kr::<Ratio<BigInt>>(link),
        }
    }

    fn compute_kr<R>(&self, link: &Link) -> KRHomologyStr
    where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
        let kr = self.init_kr(link);

        info!("compute KRHomology.");
        
        info!("i-range: {:?}", kr.i_range());
        info!("j-range: {:?}", kr.j_range());
        info!("k-range: {:?}", kr.k_range());
        info!("q-range: {:?}", kr.q_range());
        
        info!("targets:");
        kr.support().enumerate().for_each(|(i, idx)| 
            info!("  {}. {idx}", i + 1)
        );

        info!("- - - - - - - - - - - - - - - -");

        let total = kr.support().count();
        
        kr.support().enumerate().filter_map(|(i, idx)| {
            info!("({}/{}) H[{}] ..", i + 1, total, idx);

            let h = kr.get(idx);
            
            info!("H[{}] => {}", idx, h.math_symbol());
            info!("- - - - - - - - - - - - - - - -");

            if h.rank() > 0 { 
                Some(((idx.0, idx.1, idx.2), h.rank()))
            } else { 
                None
            }
        }).collect()
    }

    fn format(&self, res: &KRHomologyStr) -> String { 
        match &self.args.format {
            Output::Table =>  res.qpoly_table(),
            Output::Poly =>   res.poincare_poly().to_string(),
            Output::Homfly => res.homfly_poly().to_string(),
        }
    }

    fn make_link(&self) -> Result<Link, Box<dyn std::error::Error>> { 
        let target = &self.args.target;
        let braid = Braid::load(target)?;

        if braid.len() > MAX_BRAID_LEN { 
            err!("braid-length {} of '{target}' exceeds limit: {MAX_BRAID_LEN}", braid.len())?;
        }

        info!("braid: {}", braid);

        let link = braid.closure();
        let link = if self.args.mirror { link.mirror() } else { link };

        Ok(link)
    }

    fn init_kr<R>(&self, link: &Link) -> KRHomology<R>
    where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
        // TODO support saving. 
        let data = KRCubeData::<R>::new(&link, 2);
        KRHomology::<R>::from_data(data)
    }
}