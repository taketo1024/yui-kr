use std::sync::Arc;

use itertools::Itertools;
use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, isize3};
use yui_link::Link;

use crate::{KRHomologyStr, KRHomology};
use crate::kr::data::KRCubeData;

pub struct KRCalc<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    name: String,
    data: Arc<KRCubeData<R>>,
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
        let data = Arc::new(KRCubeData::new(&l));
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
            std::fs::create_dir(&path)?;
        }
        Ok(())
    }

    pub fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> { 
        if self.save_progress { 
            self.prepare_dir()?;
            File::Result(&self.name).write(&self.result)?;
        }

        let total = self.result.non_determined().count();

        info!("compute KRHomology.");
        info!("targets: {total}");
        
        let targets = self.q_sliced_targets();

        let q_total = targets.len();
        let mut c = 1;

        for (q, list) in targets { 
            info!("- - - - - - - - - - - - - - - -");
            info!("({c}/{q_total}) q: {q}");

            self.compute_in(q, &list)?;

            c += 1;
        }

        info!("- - - - - - - - - - - - - - - -");

        Ok(())
    }

    fn compute_in(&mut self, q: isize, targets: &Vec<isize3>) -> Result<(), Box<dyn std::error::Error>> { 
        let kr = KRHomology::from_data(self.data.clone());
        
        info!("range: {:?}", kr.range_for(q));

        for &idx in targets { 
            let inner = self.data.to_inner_grad(idx).unwrap();
            let h = kr.get(idx);

            info!("H[{}] (h: {}, v: {}) => {}", idx, inner.1, inner.2, h.math_symbol());

            self.result.set((idx.0, idx.1, idx.2), h.rank());

            if self.save_progress { 
                File::Result(&self.name).write(&self.result)?;
            }
        }

        Ok(())
    }

    fn q_sliced_targets(&self) -> Vec<(isize, Vec<isize3>)> {
        self.result.non_determined().map(|idx| isize3::from(*idx)).into_group_map_by(|&idx|
            self.data.to_inner_grad(idx).unwrap().0 // q
        ).into_iter().sorted_by_key(|(q_slice, _)| *q_slice).collect()
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
            File::Result(_) => "result",
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
        info!("load: {}", self.path());
        let str = std::fs::read_to_string(self.path())?;
        let data = serde_json::from_str::<D>(&str)?;
        Ok(data)
    }

    fn write<D>(&self, data: D) -> std::io::Result<()>
    where D: serde::Serialize {
        info!("save: {}", self.path());
        let json = serde_json::to_string(&data)?;
        std::fs::write(self.path(), &json)?;
        Ok(())
    }

    fn delete(&self) -> std::io::Result<()> { 
        if self.exists() { 
            info!("delete: {}", self.path());
            std::fs::remove_file(&self.path())?;
        }
        Ok(())
    }
}
