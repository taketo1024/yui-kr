use log::info;
use num_traits::Zero;
use yui::{EucRing, EucRingOps};
use yui_homology::utils::ChainReducer;
use yui_homology::{isize2, DisplayTable, ChainComplex2, GridTrait, ChainComplexTrait};
use yui_matrix::MatTrait;
use yui_matrix::sparse::SpMat;

use super::tot_cpx::KRTotComplex;

pub struct KRTotComplexReducer<'a, R>
where R: EucRing, for<'x> &'x R: EucRingOps<R> { 
    complex: &'a KRTotComplex<R>,
    reducer: ChainReducer<isize2, R>
}

impl<'a, R> KRTotComplexReducer<'a, R> 
where R: EucRing, for<'x> &'x R: EucRingOps<R> {
    pub(crate) fn new(complex: &'a KRTotComplex<R>) -> Self { 
        let reducer = ChainReducer::new(
            complex.support(), 
            complex.d_deg(), 
            false
        );
        Self { complex, reducer }
    }

    pub(crate) fn reduce(&mut self) { 
        info!("reduce C_tot (q: {})..", self.complex.q_deg());

        self.reduce_initial();

        if !self.reducer.is_done() { 
            info!("second run..");

            self.reduce_second();
        }
    }

    fn reduce_initial(&mut self) { 
        for idx in self.complex.support() {
            let to_idx = idx + self.complex.d_deg();
            
            if self.complex.rank(idx).is_zero() { 
                let m = self.complex.rank(to_idx);
                self.reducer.set_matrix(idx, SpMat::zero((m, 0)));
                continue;
            }

            info!("d {idx} -> {to_idx}");

            // MEMO: The 'next matrix' will serve as trans-back for the current one. 
            // This is to reduce the input dim without `with_trans = true`.
            
            let d = if let Some(q) = self.reducer.matrix(idx) {
                self.complex.d_matrix_for(idx, &q)
            } else { 
                self.complex.d_matrix(idx)
            };

            if self.complex.is_supported(to_idx) { 
                let m = d.nrows();
                self.reducer.set_matrix(to_idx, SpMat::id(m));
            }

            self.reducer.set_matrix(idx, d);

            self.reduce_at(idx, false);
        }
    }

    fn reduce_second(&mut self) { 
        for idx in self.complex.support() {
            let to_idx = idx + self.complex.d_deg();
            let d = self.reducer.matrix(idx).unwrap();
            
            if d.is_zero() { 
                continue;
            }

            info!("d {idx} -> {to_idx}");

            self.reduce_at(idx, false);
        }    
    }

    fn reduce_at(&mut self, idx: isize2, deep: bool) { 
        let d = self.reducer.matrix(idx).unwrap();

        info!("  size:    {:?}", d.shape());

        self.reducer.reduce_at(idx, deep);

        info!("  reduced: {:?}", self.reducer.matrix(idx).unwrap().shape());
    }

    pub(crate) fn into_complex(self) -> ChainComplex2<R> { 
        let red = self.reducer.into_complex();

        info!("reduced C (q: {})\n{}", self.complex.q_deg(), red.display_table("h", "v"));
        
        red
    }
}