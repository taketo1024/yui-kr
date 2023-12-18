use log::info;
use yui::{EucRing, EucRingOps};
use yui_homology::{GridTrait, RModStr};
use yui_link::Link;

use crate::{KRHomologyStr, KRHomology};
use crate::kr::data::KRCubeData;

pub struct KRCalc<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    data: KRCubeData<R>,
    result: KRHomologyStr
}

impl<R> KRCalc<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub fn init(l: &Link) -> Self { 
        let data = KRCubeData::new(&l);
        let result = data.support().map(|idx| ((idx.0, idx.1, idx.2), None)).collect();
        Self { data, result }
    }

    pub fn result(&self) -> &KRHomologyStr { 
        &self.result
    }

    pub fn into_result(self) -> KRHomologyStr { 
        self.result
    }

    pub fn compute(&mut self) { 
        let kr = KRHomology::from_data(self.data.clone());

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
        self.result = kr.support().enumerate().filter_map(|(i, idx)| {
            info!("({}/{}) H[{}] ..", i + 1, total, idx);

            let h = kr.get(idx);
            
            info!("H[{}] => {}", idx, h.math_symbol());
            info!("- - - - - - - - - - - - - - - -");

            if h.rank() > 0 { 
                Some(((idx.0, idx.1, idx.2), h.rank()))
            } else { 
                None
            }
        }).collect();
    }
}