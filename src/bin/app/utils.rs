use super::app::{AppErr, err};
use log::info;
use yui_kr::KRHomologyStr;

const RESULT_DIR: &str = "results";

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
    File::Result(name).exists()
}

pub fn load_result(name: &str) -> Result<KRHomologyStr, Box<dyn std::error::Error>> {
    File::Result(name).read()
}

pub fn save_result(name: &str, data: &KRHomologyStr) -> Result<(), Box<dyn std::error::Error>> { 
    let file = File::Result(name);

    if let Ok(prev) = load_result(name) { 
        if &prev == data { 
            info!("data exists: {}", file.path());
            return Ok(())
        } else { 
            info!("save (overwrite): {}", file.path());
        }
    } else { 
        info!("save: {}", file.path());
    }

    file.write(data)?;

    Ok(())
}

enum File<'a> { 
    Result(&'a str)
}

impl<'a> File<'a> { 
    fn dir(&self) -> String { 
        let proj_dir = std::env!("CARGO_MANIFEST_DIR");
        let dir = match self {
            File::Result(_) => RESULT_DIR
        };
        format!("{proj_dir}/{dir}")
    }

    fn path(&self) -> String { 
        let dir = self.dir();
        let name = match self {
            File::Result(name) => name,
        };
        let ext = "json";
        format!("{dir}/{name}.{ext}")
    }

    fn exists(&self) -> bool { 
        std::path::Path::new(&self.path()).exists()
    }
    
    fn read<D>(&self) -> Result<D, Box<dyn std::error::Error>>
    where for<'de> D: serde::Deserialize<'de> { 
        let str = std::fs::read_to_string(&self.path())?;
        let data = serde_json::from_str::<D>(&str)?;
        Ok(data)
    }
    
    fn write<D>(&self, data: D) -> std::io::Result<()>
    where D: serde::Serialize {
        let json = serde_json::to_string(&data)?;
        std::fs::write(&self.path(), &json)?;
        Ok(())
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
    fn load() { 
        let res = load_result("3_1");
        
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), KRHomologyStr::from(hashmap!{ 
            (0,4,-2) => 1,
            (-2,2,2) => 1,
            (2,2,-2) => 1
        }));
    }
}