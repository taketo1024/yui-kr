#[path = "../src/bin/app/mod.rs"]
pub mod app;

use log::info;

fn check_homfly(target: &str) -> Result<(), Box<dyn std::error::Error>>  {
    use app::utils::*;

    let proj_dir = std::env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{proj_dir}/data/");
    let mut reader = csv::Reader::from_path(&format!("{data_dir}/{target}.csv"))?;

    for r in reader.records() { 
        let r = r?;
        let name = &r[0];

        if !result_exists(name) { 
            info!("skip: {name}");
            continue
        }

        let data = &r[1];
        let data = data.replace(";", ",");
        let answer = yui_kr::util::parse_homfly(&data)?;

        let result = load_result(name)?;
        let poly = yui_kr::util::homfly_poly(&result);

        assert_eq!(answer, poly);
    }

    Ok(())
}

#[test]
fn check_homfly_10() -> Result<(), Box<dyn std::error::Error>> { 
    check_homfly("homfly-10")
}

#[test]
fn check_homfly_11() -> Result<(), Box<dyn std::error::Error>> { 
    check_homfly("homfly-11")
}