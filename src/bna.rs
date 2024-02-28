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

pub fn quarter(vneuron: &VNeuron) -> f64 {
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS - 1 {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = vneuron.evaluate(&vec![1., angle]);
        if output && angle <= PI / 2. || !output && angle > PI / 2. {
            sum += 1;
        }
    }

    sum as f64 / (UNIT_CIRCLE_STEPS - 1) as f64
}

pub fn two_quarters(vneuron: &VNeuron) -> f64 {
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS - 1 {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = vneuron.evaluate(&vec![1., angle]);
        if output && (angle <= PI / 2. || angle >= PI && angle <= 3. * PI / 2.)
        || !output && (angle > PI / 2. && angle < PI || angle > 3. * PI / 2.) {
            sum += 1;
        }
    }

    sum as f64 / (UNIT_CIRCLE_STEPS - 1) as f64
}

pub fn square(vneuron: &VNeuron) -> f64 {
    let points_with_labels = [
        (1., PI / 4., true),
        (1., 3. * PI / 4., false),
        (1., 5. * PI / 4., true),
        (1., 7. * PI / 4., false),
    ];

    points_with_labels
        .iter()
        .map(|&(r, theta, label)| {
            let output = vneuron.evaluate(&vec![r, theta]);
            if output && label || !output && !label {
                1.
            } else {
                0.
            }
        })
        .sum::<f64>() / 4.
}
