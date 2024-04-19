use std::f64::consts::PI;
use neuroevolution::pole_balancing::State;

fn main() {
    let mut state = State::new(
        0.,
        0.,
        vec![1.],
        vec![1. * PI / 3.],
        vec![0.],
        1.,
        vec![0.5],
    );

    let force = 0.;
    println!("{:?}", state);
    for _ in 0..1000000 {
        state.update(force);
    }
    println!("{:?}", state);
}
