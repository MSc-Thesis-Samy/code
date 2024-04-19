use std::f64::consts::PI;
use neuroevolution::pole_balancing::State;

fn main() {
    let mut state = State::new(
        0.,
        0.,
        vec![1.],
        vec![PI],
        vec![0.],
        1000000.,
        vec![5.],
    );

    let force = 0.;
    println!("{:?}", state);
    for _ in 0..1000 {
        state.update(force);
    }
    println!("{:?}", state);
}
