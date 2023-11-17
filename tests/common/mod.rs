use std::collections::HashMap;

pub type Res = HashMap<(isize, isize, isize), usize>;

pub fn load_data(name: &str) -> Result<Res, Box<dyn std::error::Error>> {
    const RESULT_DIR: &str = "results/raw/";

    let dir = std::env!("CARGO_MANIFEST_DIR");
    let path = format!("{dir}/{RESULT_DIR}{name}.json");
    let json = std::fs::read_to_string(path)?;
    let list: Vec<((isize, isize, isize), usize)> = serde_json::from_str(&json)?;
    let data = list.into_iter().collect();

    Ok(data)
}

macro_rules! test {
    ($test:ident, $name:literal) => {
        #[test]
        #[ignore]
        fn $test() { 
            use yui::Ratio;
            use yui_link::Braid;
            use yui_kr::kr::KRHomology;

            type R = Ratio<i128>;

            let b = Braid::load($name).unwrap();
            let l = b.closure();
            let kr = KRHomology::<R>::new(&l);
            let str = kr.rank_all();
            let ans = common::load_data($name).expect("");
            assert_eq!(str, ans);
        }                    
    };
}

pub(crate) use test;