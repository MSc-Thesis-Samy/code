use neuroevolution::one_plus_one_na::*;
use std::f64::consts::PI;

const STEPS: u32 = 1000;

fn half<const N: usize>(network: &Network<N>) -> f64 {
    let mut sum = 0;
    for i in 0..STEPS {
        let angle = 2. * PI * i as f64 / STEPS as f64;
        let output = network.evaluate((1., angle));
        if output && angle < PI || !output && angle > PI {
            sum += 1;
        }
    }
    sum as f64 / STEPS as f64
}

fn main() {
    let mut network = Network::<1>::new();
    network.optimize(half, 1000);
    println!("{}", network);
    println!("Fitness: {}", half(&network));
}
