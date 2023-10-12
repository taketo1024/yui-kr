use yui_core::{EucRing, EucRingOps};
use yui_link::Link;

use super::data::KRCubeData;

struct KRHomology<R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    data: KRCubeData<R>
}

impl<R> KRHomology<R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    pub fn new(link: &Link) -> Self { 
        let data = KRCubeData::new(&link);
        Self { data }
    }
}