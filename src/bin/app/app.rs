use log::{info, error};
use clap::{Parser, ValueEnum};
use derive_more::Display;
use num_bigint::BigInt;
use yui::{Ratio, EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr};
use yui_kr::kr::data::KRCubeData;
use yui_kr::{KRHomology, KRHomologyStr};
use yui_kr::util::{make_qpoly_table, mirror};
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
        let output = make_qpoly_table(&res);

        info!("time: {:?}", time);

        Ok(output)
    }

    fn compute(&self) -> Result<KRHomologyStr, Box<dyn std::error::Error>> { 
        if !self.args.force_compute && Self::data_exists(&self.args.target) { 
            info!("result exists: {}", self.args.target);
            let res = Self::load_data(&self.args.target)?;
            let res = if self.args.mirror { mirror(&res) } else { res };
            return Ok(res)
        }
        
        let link = self.make_link()?;

        let res = if link.writhe() > 0 { 
            info!("compute from mirror.");
            let link = link.mirror();
            let res = self.compute_kr_dispatch(&link);
            mirror(&res)
        } else { 
            self.compute_kr_dispatch(&link)
        };

        if self.args.check_result { 
            Self::check_result(&self.args.target, &res)?;
        }
        
        if self.args.save_result { 
            Self::save_data(&self.args.target, &res)?;
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

    pub fn path_for(name: &str) -> String { 
        const RESULT_DIR: &str = "results/raw/";
        let dir = std::env!("CARGO_MANIFEST_DIR");
        format!("{dir}/{RESULT_DIR}{name}.json")
    }

    pub fn data_exists(name: &str) -> bool { 
        let path = Self::path_for(name);
        std::path::Path::new(&path).exists()
    }

    pub fn load_data(name: &str) -> Result<KRHomologyStr, Box<dyn std::error::Error>> {
        let path = Self::path_for(name);
        let json = std::fs::read_to_string(path)?;
        let list: Vec<((isize, isize, isize), usize)> = serde_json::from_str(&json)?;
        let data = list.into_iter().collect();
        Ok(data)
    }

    pub fn save_data(name: &str, data: &KRHomologyStr) -> Result<(), Box<dyn std::error::Error>> { 
        if Self::data_exists(name) { 
            let exist = Self::load_data(name)?;
            if data == &exist { 
                info!("data already exists: {}", Self::path_for(name));
                return Ok(())
            }
            info!("overwrite existing result: {}", Self::path_for(name));
        } else { 
            info!("save: {}", Self::path_for(name));
        }

        let path = Self::path_for(name);
        let json = serde_json::to_string(&data.iter().collect::<Vec<_>>())?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn check_result(name: &str, data: &KRHomologyStr) -> Result<(), Box<dyn std::error::Error>> {
        if Self::data_exists(name) { 
            let expected = Self::load_data(name)?;
            if data != &expected { 
                err!("Incorrect result for {name}.\nComputed: {data:#?},\nExpected: {expected:#?}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests { 
    use yui::hashmap;

    use super::*;

    #[test]
    fn data_exists() { 
        assert!( App::data_exists("3_1"));
        assert!(!App::data_exists("3_2"));
    }

    #[test]
    fn check_result_ok() { 
        let data = hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        };
        let res = App::check_result("3_1", &data);
        assert!(res.is_ok());
    }

    #[test]
    fn check_result_ng() { 
        let data = hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 2,
            (2,2,-2) => 1
        };
        let res = App::check_result("3_1", &data);
        assert!(res.is_err());
    }
}