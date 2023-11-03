use yui_kr::kr::*;
use yui_link::Link;
use yui_ratio::Ratio;

fn main() {
    measure(|| { 
        run();
    })
}

fn run() { 
    type R = Ratio<i64>;

    let l = Link::trefoil();
    let h = KRHomology::<R>::new(&l);
    h.print_table();
}

fn measure<F, Res>(proc: F) -> Res
where F: FnOnce() -> Res { 
    let start = std::time::Instant::now();
    let res = proc();
    let time = start.elapsed();

    println!("time: {:?}", time);

    res
}