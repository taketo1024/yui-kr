use std::sync::Arc;

use itertools::Itertools;
use log::info;
use yui_core::{EucRing, EucRingOps};
use yui_homology::{GridTrait, SummandTrait, isize3};
use yui_link::Link;

use yui_kr::internal::data::{KRCubeData, range2};
use yui_kr::internal::tot_homol::KRTotHomol;
use yui_kr::KRHomologyStr;

#[derive(Clone, Copy, Default, Debug, clap::ValueEnum)]
pub enum KRCalcMode {
    #[default] Default, PerCol, PerItem
}

pub struct KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    result: KRHomologyStr,
    pub mode: KRCalcMode,
    pub size_limit: usize,
}

impl<R> KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn init(l: &Link) -> Self {
        let data = Arc::new(KRCubeData::new(l));
        let result = data.support().map(|&idx| ((idx.0, idx.1, idx.2), None)).collect();
        let mode = KRCalcMode::default();
        let size_limit = usize::MAX;
        Self { data, result, mode, size_limit }
    }

    pub fn result(&self) -> &KRHomologyStr {
        &self.result
    }

    pub fn into_result(self) -> KRHomologyStr {
        self.result
    }

    pub fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
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