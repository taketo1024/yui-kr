use yui_kr::kr::*;
use yui_link::Link;
use yui::Ratio;

fn main() {
    measure("run", || { 
        run();
    })
}

fn run() { 
    type R = Ratio<i64>;

    let l = Link::trefoil();
    let h = KRHomology::<R>::new(&l);
    h.print_table();
}

fn measure<F, Res>(name: &str, proc: F) -> Res
where F: FnOnce() -> Res { 
    println!("{name} start...\n");
    
    let start = std::time::Instant::now();
    let res = proc();
    let time = start.elapsed();

    println!("{name} time: {:?}", time);

    res
}