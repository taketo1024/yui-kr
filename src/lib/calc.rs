use std::sync::Arc;

use itertools::Itertools;
use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, isize3};
use yui_link::Link;

use crate::{KRHomologyStr, KRHomology};
use crate::internal::data::KRCubeData;

pub struct KRCalc<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    name: String,
    data: Arc<KRCubeData<R>>,
    result: KRHomologyStr,
    pub save_progress: bool
}

impl<R> KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
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

        info!("compute KRHomology.");
        info!("q-range: {:?}", self.data.q_range());
        
        let targets = self.organize_targets();
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
        let kr = KRHomology::new_restr(self.data.clone(), q..=q);
        
        for &idx in targets { 
            let h = kr.get(idx);
            self.result.set((idx.0, idx.1, idx.2), h.rank());
        }

        if self.save_progress { 
            File::Result(&self.name).write(&self.result)?;
        }

        Ok(())
    }

    fn organize_targets(&self) -> Vec<(isize, Vec<isize3>)> {
        self.result.non_determined().map(|&idx| {
            // convert to inner-grad.
            let outer = idx.into();
            let inner = self.data.to_inner_grad(outer).unwrap();
            (outer, inner)
        }).into_group_map_by(|(_, inner)|
            // group by q.
            inner.0
        ).into_iter().map(|(q, list)| { 
            // order by lex in (h, v)
            let list = list.into_iter().sorted_by(|e, f| { 
                isize::cmp(&e.1.1, &f.1.1).then( isize::cmp(&e.1.2, &f.1.2) )
            }).map(|(outer, _)| 
                outer
            ).collect_vec();
            (q, list)
        }).sorted_by_key(|(q, _)| *q).collect()
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
