#![allow(unused)]

use super::app::{AppErr, err};
use std::str::FromStr;
use log::info;
use num_traits::Zero;
use yui_kr::KRHomologyStr;
use yui_link::{Link, Edge};

const RESULT_DIR: &str = "results";
const TMP_DIR: &str = "tmp";

pub fn measure<F, Res>(proc: F) -> (Res, std::time::Duration) 
where F: FnOnce() -> Res { 
    let start = std::time::Instant::now();
    let res = proc();
    let time = start.elapsed();
    (res, time)
}

pub fn guard_panic<F, R>(f: F) -> Result<R, Box<dyn std::error::Error>>
where F: FnOnce() -> Result<R, Box<dyn std::error::Error>> + std::panic::UnwindSafe {
    std::panic::catch_unwind(|| {
        f()
    }).unwrap_or_else(|e| {
        let info = match e.downcast::<String>() {
            Ok(v) => *v,
            Err(e) => match e.downcast::<&str>() {
                Ok(v) => v.to_string(),
                _ => "Unknown Source of Error".to_owned()
            }
        };
        err!("panic: {info}")
    })
}

pub fn result_exists(name: &str) -> bool {
    let file = File::Result(name);
    file_exists(&file)
}

pub fn load_result(name: &str) -> Result<KRHomologyStr, Box<dyn std::error::Error>> {
    let file = File::Result(name);
    let data = read_json(&file)?;
    let data = deserialize(&data);
    Ok(data)
}

pub fn save_result(name: &str, data: &KRHomologyStr) -> Result<(), Box<dyn std::error::Error>> { 
    let file = File::Result(name);
    if file_exists(&file) { 
        info!("overwrite existing result: {}", file.path());
    } else { 
        info!("save: {}", file.path());
    }

    let data = serialize(&data);
    write_json(&file, data)?;
    Ok(())
}

pub fn check_result(name: &str, data: &KRHomologyStr) -> Result<(), Box<dyn std::error::Error>> {
    if result_exists(name) { 
        let expected = load_result(name)?;
        if data != &expected { 
            err!("Incorrect result for {name}.\nComputed: {data:#?},\nExpected: {expected:#?}")?;
        }
    }
    Ok(())
}

fn file_exists(file: &File) -> bool { 
    std::path::Path::new(&file.path()).exists()
}

fn read_string(file: &File) -> std::io::Result<String> { 
    std::fs::read_to_string(file.path())
}

fn write_string(file: &File, str: &str) -> std::io::Result<()> { 
    std::fs::write(file.path(), str)
}

fn read_json<D>(file: &File) -> Result<D, Box<dyn std::error::Error>>
where for<'de> D: serde::Deserialize<'de> { 
    let str = read_string(file)?;
    let data = serde_json::from_str::<D>(&str)?;
    Ok(data)
}

fn write_json<D>(file: &File, data: D) -> std::io::Result<()>
where D: serde::Serialize {
    let json = serde_json::to_string(&data)?;
    write_string(file, &json)?;
    Ok(())
}

fn serialize(str: &KRHomologyStr) -> Vec<(isize, isize, isize, usize)> { 
    str.iter().map(|((i, j, k), r)| (*i, *j, *k, *r)).collect()
}

fn deserialize(str: &Vec<(isize, isize, isize, usize)>) -> KRHomologyStr { 
    str.iter().map(|(i, j, k, r)| ((*i, *j, *k), *r)).collect()
}

#[allow(unused)]
enum File<'a> { 
    Result(&'a str), 
    Tmp(&'a str)
}

impl<'a> File<'a> { 
    fn dir(&self) -> String { 
        let proj_dir = std::env!("CARGO_MANIFEST_DIR");
        let dir = match self {
            File::Result(_) => RESULT_DIR,
            File::Tmp(_) => TMP_DIR,
        };
        format!("{proj_dir}/{dir}")
    }

    fn path(&self) -> String { 
        let dir = self.dir();
        let name = match self {
            File::Result(name) | File::Tmp(name) => name,
        };
        let ext = "json";
        format!("{dir}/{name}.{ext}")
    }
}

#[cfg(test)]
#[cfg(not(feature = "batch_test"))]
mod tests { 
    use yui::hashmap;
    use super::*;

    #[test]
    fn data_exists() { 
        assert!( result_exists("3_1"));
        assert!(!result_exists("3_2"));
    }

    #[test]
    fn check_result_ok() { 
        let data = KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        });
        let res = check_result("3_1", &data);
        assert!(res.is_ok());
    }

    #[test]
    fn check_result_ng() { 
        let data = KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 2,
            (2,2,-2) => 1
        });
        let res = check_result("3_1", &data);
        assert!(res.is_err());
    }
}