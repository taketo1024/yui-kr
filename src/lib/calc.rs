//! [`KRCalc`]: the same computation as [`KRHomology`](crate::KRHomology), but
//! run grading by grading so that it can be bounded and resumed.

use std::sync::Arc;

use cartesian::cartesian;
use itertools::Itertools;
use log::info;
use yui_core::abst::{EucRing, EucRingOps};
use yui_homology::{isize2, isize3};
use yui_link::Link;

use crate::internal::data::{KRCubeData, range2};
use crate::internal::tot_homol::KRTotHomol;
use crate::KRHomologyStr;

/// How finely the targets are grouped before each complex is built.
///
/// Coarser groups share work between gradings; finer ones keep each complex
/// smaller, which matters when memory is the binding constraint.
#[derive(Clone, Copy, Default, Debug, clap::ValueEnum)]
pub enum KRCalcMode {
    /// One complex per `q`-slice.
    #[default] Default,
    /// One per `(q, h)`.
    PerCol,
    /// One per `(q, h, v)`.
    PerItem
}

/// An incremental computation of `ℋ(K)`.
///
/// Unlike [`KRHomology`](crate::KRHomology), the result is built up grading by
/// grading into a [`KRHomologyStr`], so a run can be capped by
/// [`size_limit`](Self::size_limit) and continued later — gradings that were
/// skipped stay recorded as non-determined rather than as zero.
pub struct KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: Arc<KRCubeData<R>>,
    result: KRHomologyStr,
    /// Grouping of targets; see [`KRCalcMode`].
    pub mode: KRCalcMode,
    /// Skip any differential whose matrix exceeds this many rows or columns.
    /// Gradings that depend on a skipped matrix are left non-determined.
    pub size_limit: usize,
}

impl<R> KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    /// Sets up the computation, with every supported grading non-determined.
    pub fn init(l: &Link) -> Self {
        let data = Arc::new(KRCubeData::new(l));
        let result = data.support().map(|&idx| ((idx.0, idx.1, idx.2), None)).collect();
        let mode = KRCalcMode::default();
        let size_limit = usize::MAX;
        Self { data, result, mode, size_limit }
    }

    /// The result so far, including any gradings still non-determined.
    pub fn result(&self) -> &KRHomologyStr {
        &self.result
    }

    pub fn into_result(self) -> KRHomologyStr {
        self.result
    }

    /// Computes every grading still non-determined, in `q`-order.
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
        let range = range2(hv);

        let complex = KRTotHomol::make_cpx(self.data.clone(), q, range.clone());
        let complex = complex.reduced_with_limit(self.size_limit);
        let d_deg = complex.d_deg();
        
        let support = cartesian!(
            range.0.clone(), 
            range.1.clone()
        ).map(isize2::from);

        for idx in support { 
            if
                complex[idx - d_deg].rank() <= self.size_limit &&
                complex[idx].rank()         <= self.size_limit &&
                complex[idx + d_deg].rank() <= self.size_limit
            {
                let res = complex.generic_homology_at(idx);
                let grad = self.data.to_outer_grad(isize3(q, idx.0, idx.1));
                self.result.set(grad.into(), res.rank());
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
