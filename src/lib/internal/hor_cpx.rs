//! [`KRHorComplex`]: the horizontal cube of [`hor_cube`](super::hor_cube) with
//! the exclusion of [`hor_excl`](super::hor_excl) already applied.
//!
//! Only the generators surviving the exclusion are kept, and the differential
//! is `d_H` followed by reduction modulo the excluded variables.

use std::ops::{Index, RangeInclusive};
use std::sync::Arc;
use delegate::delegate;

use num_traits::Zero;
use yui_core::abst::{Ring, RingOps};
use yui_homology::{ChainComplex1, Summand, GrMod1};
use yui_core::bitseq::BitSeq;

// use log::info;
// use yui_homology::DisplaySeq;

use super::base::{KRGen, KRChain};
use super::data::KRCubeData;
use super::hor_cube::KRHorCube;
use super::hor_excl::KRHorExcl;

pub struct KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    q: isize,
    excl: Arc<KRHorExcl<R>>,
    inner: ChainComplex1<KRGen, R>
} 

impl<R> KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> { 
    pub fn new(data: Arc<KRCubeData<R>>, q: isize, v_coords: BitSeq) -> Self { 
        let n = data.dim() as isize;
        Self::new_restr(data, q, v_coords, 0..=n)
    }

    pub fn new_restr(data: Arc<KRCubeData<R>>, q: isize, v_coords: BitSeq, h_range: RangeInclusive<isize>) -> Self { 
        // info!("C_hor (q: {}, v: {:?})..", q, v_coords);

        let excl = data.excl(v_coords);
        let cube = KRHorCube::new(data.clone(), q, v_coords);
        let inner = Self::make_cpx(excl.clone(), cube, h_range);

        // info!("C_hor (q: {}, v: {:?})\n{}", q, v_coords, inner.display_seq("h"));

        Self { q, excl, inner }
    }

    fn make_cpx(excl: Arc<KRHorExcl<R>>, cube: KRHorCube<R>, h_range: RangeInclusive<isize>) -> ChainComplex1<KRGen, R> { 
        let summands = GrMod1::generate(h_range.clone(), |i| { 
            let gens = cube.gens(i as usize).filter(|v| 
                excl.should_remain(v)
            );
            Summand::from_raw_generators(gens)
        });
        
        ChainComplex1::new(summands, 1, move |i, e| {
            if i < *h_range.end() { 
                excl.d(e)
            } else { 
                KRChain::zero()
            }
        })
    }

    pub fn reduced(&self) -> ChainComplex1<KRGen, R> {
        // info!("reduce C_hor (q: {}, v: {:?})..", self.q, self.v_coords);

        let red = self.inner.reduced();

        // info!("reduce C_hor (q: {}, v: {:?})\n{}", self.q, self.v_coords, red.display_seq("h"));

        red
    }

    pub fn q_deg(&self) -> isize {
        self.q
    }

    pub fn v_coords(&self) -> BitSeq {
        self.excl.v_coords()
    }

    delegate! {
        to self.inner {
            pub fn support(&self) -> impl Iterator<Item = &isize> + '_;
            pub fn is_supported(&self, i: isize) -> bool;
            pub fn d_deg(&self) -> isize;
        }
    }

    pub fn d(&self, i: isize, z: &KRChain<R>) -> KRChain<R> {
        let z_exc = self.excl.forward(z);
        let dz_exc = self.inner.d(i, &z_exc);
        let dz = self.excl.backward(&dz_exc, true);
        dz
    }
}

impl<R> Index<isize> for KRHorComplex<R>
where R: Ring, for<'x> &'x R: RingOps<R> {
    type Output = Summand<KRGen, R>;

    fn index(&self, i: isize) -> &Self::Output {
        &self.inner[i]
    }
}

#[cfg(test)]
mod tests { 
        use yui_link::Link;
    use yui_core::num::Ratio;

    use super::*;

    type R = Ratio<i64>;

    fn make_cpx(link: &Link, v_coords: BitSeq, q: isize, level: usize, red: bool) -> KRHorComplex<R> {
        let n = link.n_crossings() as isize;
        let data = Arc::new( KRCubeData::<R>::new_excl(link, level) );
        let excl = data.excl(v_coords);
        let cube = KRHorCube::new(data.clone(), q, v_coords);
        let inner = KRHorComplex::make_cpx(excl.clone(), cube, 0..=n);
        let inner = if red { 
            inner.reduced()
        } else { 
            inner
        };

        KRHorComplex { q, excl, inner }
    }

    #[test]
    fn no_red() { 
        let l = Link::test_data("3_1");
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 0, false);

        assert_eq!(c[0].rank(), 3);
        assert_eq!(c[1].rank(), 26);
        assert_eq!(c[2].rank(), 51);
        assert_eq!(c[3].rank(), 28);

        c.inner.check_d_all();

        let h = c.inner.homology();
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level1() { 
        let l = Link::test_data("3_1");
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 1, false);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 3);
        assert_eq!(c[2].rank(), 10);
        assert_eq!(c[3].rank(), 7);

        c.inner.check_d_all();

        let h = c.inner.homology();
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level2() { 
        let l = Link::test_data("3_1");
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 2, false);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.inner.check_d_all();

        let h = c.inner.homology();
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level1_red() { 
        let l = Link::test_data("3_1");
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 1, true);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.inner.check_d_all();

        let h = c.inner.homology();
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }

    #[test]
    fn excl_level2_red() { 
        let l = Link::test_data("3_1");
        let v = BitSeq::from([1,0,0]);
        let q = 0;
        
        let c = make_cpx(&l, v, q, 2, true);

        assert_eq!(c[0].rank(), 0);
        assert_eq!(c[1].rank(), 0);
        assert_eq!(c[2].rank(), 2);
        assert_eq!(c[3].rank(), 2);

        c.inner.check_d_all();

        let h = c.inner.homology();
        
        assert_eq!(h[0].rank(), 0);
        assert_eq!(h[1].rank(), 0);
        assert_eq!(h[2].rank(), 2);
        assert_eq!(h[3].rank(), 2);
    }
}