use std::f64::consts::PI;
use cmaes::DVector;
use crate::network::*;

const UNIT_CIRCLE_STEPS: u32 = 1000;

pub fn half<const N: usize>(x: &DVector<f64>) -> f64 {
    if x.iter().all(|&x| x >= 0. && x <= 1.) {
        let network: Network<N,2> = get_network(x);
        half_aux(&network)
    } else {
        f64::NEG_INFINITY
    }
}

pub fn half_aux<const N: usize>(network: &Network<N,2>) -> f64 {
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = network.evaluate(&[1., angle]);
        if output && angle < PI || !output && angle > PI {
            sum += 1;
        }
    }
    sum as f64 / UNIT_CIRCLE_STEPS as f64
}
