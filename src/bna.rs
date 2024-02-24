use std::f64::consts::PI;
use crate::vneuron::*;

const UNIT_CIRCLE_STEPS: u32 = 100;

pub fn half(vneuron: &VNeuron) -> f64 {
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS - 1 {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = vneuron.evaluate(&vec![1., angle]);
        if output && angle <= PI || !output && angle > PI {
            sum += 1;
        }
    }

    sum as f64 / (UNIT_CIRCLE_STEPS - 1) as f64
}
