use yui_kr::kr::*;
use yui_link::Link;
use yui_ratio::Ratio;

fn main() {
    type R = Ratio<i64>;
    
    let l = Link::trefoil();
    let h = KRHomology::<R>::new(&l);

    h.print_table();
}
