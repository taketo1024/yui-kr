use std::sync::Arc;

use itertools::Itertools;
use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr, isize3};
use yui_link::Link;

use crate::internal::tot_homol::KRTotHomol;
use crate::KRHomologyStr;
use crate::internal::data::{KRCubeData, range2};

#[derive(Clone, Copy, Default, Debug, clap::ValueEnum)]
pub enum KRCalcMode { 
    #[default] Default, PerCol, PerItem
}

pub struct KRCalc<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    name: String,
    data: Arc<KRCubeData<R>>,
    result: KRHomologyStr,
    pub mode: KRCalcMode,
    pub size_limit: usize,
    pub save_progress: bool
}

impl<R> KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn init(name: &str, l: &Link) -> Self { 
        let name = name.to_owned();
        let data = Arc::new(KRCubeData::new(&l));
        let result = data.support().map(|idx| ((idx.0, idx.1, idx.2), None)).collect();
        let mode = KRCalcMode::default();
        let size_limit = usize::MAX;
        let save_progress = false;
        Self { name, data, result, mode, size_limit, save_progress }
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
        
        let targets = match self.mode {
            KRCalcMode::Default => self.group_targets_q(),
            KRCalcMode::PerCol  => self.group_targets_qh(),
            KRCalcMode::PerItem => self.group_targets_qhv(),
        };
        let q_total = targets.len();

        for (c, (q, list)) in targets.iter().enumerate() { 
            info!("({}/{q_total}) - - - - - - - - - - - -", c + 1);
            self.compute_in(*q, list)?;
        }

        info!("- - - - - - - - - - - - - - - -");

        Ok(())
    }

    fn compute_in(&mut self, q: isize, targets: &Vec<isize3>) -> Result<(), Box<dyn std::error::Error>> { 
        let hv = targets.iter().map(|&idx| {
            let inner = self.data.to_inner_grad(idx).unwrap();
            assert_eq!(inner.0, q);
            (inner.1, inner.2)
        });
        let hv_range = range2(hv);
        let kr = KRTotHomol::try_partial(self.data.clone(), q, hv_range, self.size_limit);
        
        for &idx in targets { 
            let inner = self.data.to_inner_grad(idx).unwrap();
            if let Some(h) = kr.get((inner.1, inner.2).into()) { 
                self.result.set(idx.into(), h.rank());
            }
        }

        if self.save_progress { 
            File::Result(&self.name).write(&self.result)?;
        }

        Ok(())
    }

    fn group_targets_q(&self) -> Vec<(isize, Vec<isize3>)> {
        self.result.non_determined().map(|&idx|
            isize3::from(idx)
        ).into_group_map_by(|&outer|
            self.data.to_inner_grad(outer).unwrap().0
        ).into_iter().sorted_by_key(|(q, _)| 
            *q
        ).collect()
    }

    fn group_targets_qh(&self) -> Vec<(isize, Vec<isize3>)> {
        self.group_targets_q().into_iter().flat_map(|(q, list)| { 
            list.into_iter().into_group_map_by(|&outer|
                self.data.to_inner_grad(outer).unwrap().1
            ).into_iter().sorted_by_key(|(h, _)| 
                *h
            ).map(move |(_, list)|
                (q, list)
            )
        }).collect()
    }

    fn group_targets_qhv(&self) -> Vec<(isize, Vec<isize3>)> {
        self.group_targets_qh().into_iter().flat_map(|(q, list)| { 
            list.into_iter().into_group_map_by(|&outer|
                self.data.to_inner_grad(outer).unwrap().2
            ).into_iter().sorted_by_key(|(v, _)| 
                *v
            ).map(move |(_, list)|
                (q, list)
            )
        }).collect()
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
