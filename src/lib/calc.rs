#![allow(unused)] use itertools::Itertools;
// remove later
use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, isize3};
use yui_link::Link;

use crate::{KRHomologyStr, KRHomology};
use crate::kr::data::KRCubeData;

pub struct KRCalc<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    name: String,
    data: KRCubeData<R>,
    result: KRHomologyStr,
    pub save_progress: bool
}

impl<R> KRCalc<R>
where 
    R: EucRing, for<'x> &'x R: EucRingOps<R>,
    R: serde::Serialize + for<'de> serde::Deserialize<'de>
{
    pub fn init(name: &str, l: &Link) -> Self { 
        let name = name.to_owned();
        let data = KRCubeData::new(&l);
        let result = data.support().map(|idx| ((idx.0, idx.1, idx.2), None)).collect();
        let save_progress = false;
        Self { name, data, result, save_progress }
    }

    pub fn load_if_exists(&mut self) -> Result<(), Box<dyn std::error::Error>> { 
        let file = File::Result(&self.name);
        if file.exists() { 
            self.result = file.read()?;
        }
        Ok(())
    }

    pub fn clear(&self) -> Result<(), Box<dyn std::error::Error>> { 
        File::Result(&self.name).delete()?;
        Ok(())
    }

    pub fn result(&self) -> &KRHomologyStr { 
        &self.result
    }

    pub fn into_result(self) -> KRHomologyStr { 
        self.result
    }

    fn prepare_dir(&self) -> Result<(), Box<dyn std::error::Error>> { 
        let dir = File::working_dir();
        let path = std::path::Path::new(&dir);
        if !path.exists() { 
            std::fs::create_dir_all(&path)?;
        }
        Ok(())
    }

    pub fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> { 
        if self.save_progress { 
            self.prepare_dir();
            File::Result(&self.name).write(&self.result);
        }

        let kr = KRHomology::from_data(self.data.clone());

        info!("compute KRHomology.");
        
        info!("i-range: {:?}", kr.i_range());
        info!("j-range: {:?}", kr.j_range());
        info!("k-range: {:?}", kr.k_range());
        info!("q-range: {:?}", kr.q_range());
        
        info!("targets:");
        
        let targets = self.result.non_determined().cloned().collect_vec();
        let total = targets.len();

        targets.iter().enumerate().for_each(|(i, idx)| 
            info!("  {}. {idx:?}", i + 1)
        );

        info!("- - - - - - - - - - - - - - - -");

        for (i, idx) in targets.iter().enumerate() {
            let idx = isize3(idx.0, idx.1, idx.2);
            info!("({}/{}) H[{}] ..", i + 1, total, idx);

            let h = kr.get(idx);

            info!("H[{}] => {}", idx, h.math_symbol());
            info!("- - - - - - - - - - - - - - - -");

            self.result.set((idx.0, idx.1, idx.2), h.rank());

            if self.save_progress { 
                File::Result(&self.name).write(&self.result)?;
            }
        }

        Ok(())
    }
}

enum File<'a> { 
    Result(&'a str)
}

impl<'a> File<'a> { 
    fn working_dir() -> String { 
        let proj_dir = std::env!("CARGO_MANIFEST_DIR");
        format!("{proj_dir}/tmp")
    }

    fn suffix(&self) -> &str { 
        match self { 
            File::Result(_) => "",
        }
    }

    fn path(&self) -> String { 
        let dir = Self::working_dir();
        let name = match self {
            File::Result(name) => name,
        };
        let suffix = self.suffix();
        let ext = "json";
        format!("{dir}/{name}_{suffix}.{ext}")
    }

    fn exists(&self) -> bool { 
        std::path::Path::new(&self.path()).exists()
    }

    fn read<D>(&self) -> Result<D, Box<dyn std::error::Error>>
    where for<'de> D: serde::Deserialize<'de> { 
        let str = std::fs::read_to_string(self.path())?;
        let data = serde_json::from_str::<D>(&str)?;
        Ok(data)
    }

    fn write<D>(&self, data: D) -> std::io::Result<()>
    where D: serde::Serialize {
        let json = serde_json::to_string(&data)?;
        std::fs::write(self.path(), &json)?;
        Ok(())
    }

    fn delete(&self) -> std::io::Result<()> { 
        std::fs::remove_file(&self.path())
    }
}
