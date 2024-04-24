use std::f64::consts::PI;
use crate::constants::{POLE_BALANCING_STEPS, POLE_BALANCING_MAX_FORCE};
use crate::neuroevolution_algorithm::*;
use crate::pole_balancing::State;
use clap::ValueEnum;

pub type LabeledPoint = (Vec<f64>, f64);
pub type LabeledPoints = Vec<LabeledPoint>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Problem {
    Half,
    Quarter,
    TwoQuarters,
    Square,
    Cube,
    Xor,
    PoleBalancing,
}

#[derive(Debug)]
pub enum Benchmark {
    PoleBalancing,
    Classification(LabeledPoints),
    SphereClassification(LabeledPoints)
}

impl Benchmark {
    pub fn evaluate(&self, alg: &Algorithm) -> f64 {
        match self {
            Benchmark::PoleBalancing => pole_balancing(alg),
            Benchmark::Classification(points) | Benchmark::SphereClassification(points) => classification(alg, points),
        }
    }

    pub fn new(problem: Problem) -> Self {
        match problem {
            Problem::Half => Benchmark::SphereClassification(half()),
            Problem::Quarter => Benchmark::SphereClassification(quarter()),
            Problem::TwoQuarters => Benchmark::SphereClassification(two_quarters()),
            Problem::Square => Benchmark::SphereClassification(square()),
            Problem::Cube => Benchmark::SphereClassification(cube()),
            Problem::Xor => Benchmark::Classification(xor()),
            Problem::PoleBalancing => Benchmark::PoleBalancing,
        }
    }
}

fn classification(alg: &Algorithm, points: &LabeledPoints) -> f64 {
    let distances_sum = points
        .iter()
        .map(|(point, label)| {
            let output = alg.evaluate(point);
            (output - *label).abs()
        })
        .sum::<f64>();
    (points.len() as f64 - distances_sum) / points.len() as f64
}

fn pole_balancing(alg: &Algorithm) -> f64 {
    let mut state = State::default();
    let mut count = 0;

    for _ in 0..POLE_BALANCING_STEPS {
        let input = state.to_vec();
        let output = alg.evaluate(&input);
        let force = 2. * POLE_BALANCING_MAX_FORCE * output - POLE_BALANCING_MAX_FORCE;
        state.update(force);
        if state.are_poles_balanced() && !state.is_cart_out_of_bounds() {
            count += 1;
        } else {
            break;
        }
    }

    count as f64 / POLE_BALANCING_STEPS as f64
}

fn xor() -> LabeledPoints {
    vec![
        (vec![0., 0.], 0.),
        (vec![0., 1.], 1.),
        (vec![1., 0.], 1.),
        (vec![1., 1.], 0.),
    ]
}

fn half() -> LabeledPoints {
    (0..POLE_BALANCING_STEPS)
        .map(|i| {
            let angle = 2. * PI * i as f64 / POLE_BALANCING_STEPS as f64;
            (vec![1., angle], if angle <= PI { 1. } else { 0. })
        })
        .collect::<LabeledPoints>()
}

fn quarter() -> LabeledPoints {
    (0..POLE_BALANCING_STEPS)
        .map(|i| {
            let angle = 2. * PI * i as f64 / POLE_BALANCING_STEPS as f64;
            (vec![1., angle], if angle <= PI / 2. { 1. } else { 0. })
        })
        .collect::<LabeledPoints>()
}

fn two_quarters() -> LabeledPoints {
    (0..POLE_BALANCING_STEPS)
        .map(|i| {
            let angle = 2. * PI * i as f64 / POLE_BALANCING_STEPS as f64;
            (vec![1., angle], if angle <= PI / 2. || (angle >= PI && angle <= 3. * PI / 2.) { 1. } else { 0. })
        })
        .collect::<LabeledPoints>()
}

fn square() -> LabeledPoints {
    vec![
        (vec![1., PI / 4.], 0.),
        (vec![1., 3. * PI / 4.], 1.),
        (vec![1., 5. * PI / 4.], 0.),
        (vec![1., 7. * PI / 4.], 1.),
    ]
}

fn cube() -> LabeledPoints {
    vec![
        (vec![1., PI / 4., PI / 4.], 1.),
        (vec![1., 3. * PI / 4., PI / 4.], 0.),
        (vec![1., 5. * PI / 4., PI / 4.], 1.),
        (vec![1., 7. * PI / 4., PI / 4.], 0.),
        (vec![1., PI / 4., 3. * PI / 4.], 1.),
        (vec![1., 3. * PI / 4., 3. * PI / 4.], 0.),
        (vec![1., 5. * PI / 4., 3. * PI / 4.], 1.),
        (vec![1., 7. * PI / 4., 3. * PI / 4.], 0.),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::UNIT_CIRCLE_STEPS;
    use crate::network::Network;
    use crate::vneuron::VNeuron;
    use crate::neuroevolution_algorithm::Algorithm;

    const TOL: f64 = 5e-2;

    #[test]
    fn test_half_network() {
        let network = Network::from_parameters(
            vec![0.],
            vec![vec![PI / 2.]]
        );

        let half = Benchmark::new(Problem::Half);
        assert!((half.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2.],
            vec![vec![PI / 4.]]
        );

        let quarter = Benchmark::new(Problem::Quarter);
        assert!((quarter.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_twoquarters_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2., 2f64.sqrt() / 2.],
            vec![vec![PI / 4.], vec![5. * PI / 4.]]
        );

        let two_quarters = Benchmark::new(Problem::TwoQuarters);
        assert!((two_quarters.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_null_bias() {
        let vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI / 2.
        );

        let half = Benchmark::new(Problem::Half);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_positive_bias() {
        let vneuron = VNeuron::from_parameters(
            0.5,
            vec![PI / 2.],
            2. * PI / 3.
        );

        let half = Benchmark::new(Problem::Half);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_negative_bias() {
        let vneuron = VNeuron::from_parameters(
            -0.5,
            vec![3. * PI / 2.],
            PI / 3.
        );

        let half = Benchmark::new(Problem::Half);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_vneuron_oneplusonena_solution() {
        let vneuron = VNeuron::from_parameters(
            2f64.sqrt() / 2.,
            vec![PI / 4.],
            PI / 2.
        );

        let quarter = Benchmark::new(Problem::Quarter);
        assert!((quarter.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }
}
