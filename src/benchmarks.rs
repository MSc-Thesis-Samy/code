use std::f64::consts::PI;
use crate::constants::POLE_BALANCING_STEPS;
use crate::neuroevolution_algorithm::*;
use crate::pole_balancing::State;

pub type LabeledPoint = (Vec<f64>, f64);
pub type LabeledPoints = Vec<LabeledPoint>;

#[derive(Debug)]
pub enum SphereClassificationProblem {
    Half(u32),
    Quarter(u32),
    TwoQuarters(u32),
    Square,
    Cube,
}

#[derive(Debug)]
pub enum ClassificationProblem {
    SphereProblem(SphereClassificationProblem),
    Xor,
}

fn pole_balancing(alg: &Algorithm) -> f64 {
    let mut state = State::new(
        0.,
        0.,
        vec![1.],
        vec![0.],
        vec![0.],
        1.,
        vec![0.5],
    );

    let mut count = 0;

    for _ in 0..POLE_BALANCING_STEPS {
        let input = state.to_vec();
        let output = alg.evaluate(&input);
        let force = 20. * output - 10.;
        state.update(force);
        if state.are_poles_balanced() && !state.is_cart_out_of_bounds() {
            count += 1;
        } else {
            break;
        }
    }

    count as f64 / POLE_BALANCING_STEPS as f64
}

pub trait ClassificationProblemEval {
    fn get_points(&self) -> LabeledPoints;
    fn evaluate(&self, alg: &Algorithm) -> f64 {
        match alg {
            Algorithm::Neat(neat) => {
                neat.get_best_individual_fitness()
            }
            _ => {
                let points = self.get_points();
                let distances_sum = points
                    .iter()
                    .map(|(point, label)| {
                        let output = alg.evaluate(point);
                        (output - *label).abs()
                    })
                    .sum::<f64>();
                (points.len() as f64 - distances_sum) / points.len() as f64
            }
        }
    }
}

impl ClassificationProblemEval for ClassificationProblem {
    fn get_points(&self) -> LabeledPoints {
        match self {
            ClassificationProblem::SphereProblem(problem) => problem.get_points(),
            ClassificationProblem::Xor => vec![
                (vec![0., 0.], 0.),
                (vec![0., 1.], 1.),
                (vec![1., 0.], 1.),
                (vec![1., 1.], 0.),
            ]
        }
    }
}

impl ClassificationProblemEval for SphereClassificationProblem {
    fn get_points(&self) -> LabeledPoints {
        match self {
            SphereClassificationProblem::Half(n) => {
                (0..*n)
                    .map(|i| {
                        let angle = 2. * PI * i as f64 / *n as f64;
                        (vec![1., angle], if angle <= PI { 1. } else { 0. })
                    })
                    .collect::<LabeledPoints>()
            }
            SphereClassificationProblem::Quarter(n) => {
                (0..*n)
                    .map(|i| {
                        let angle = 2. * PI * i as f64 / *n as f64;
                        (vec![1., angle], if angle <= PI / 2. { 1. } else { 0. })
                    })
                    .collect::<LabeledPoints>()
            }
            SphereClassificationProblem::TwoQuarters(n) => {
                (0..*n)
                    .map(|i| {
                        let angle = 2. * PI * i as f64 / *n as f64;
                        (vec![1., angle], if angle <= PI / 2. || angle >= 3. * PI / 2. { 1. } else { 0. })
                    })
                    .collect::<LabeledPoints>()
            }
            SphereClassificationProblem::Square => {
                vec![
                    (vec![1., PI / 4.], 0.),
                    (vec![1., 3. * PI / 4.], 1.),
                    (vec![1., 5. * PI / 4.], 0.),
                    (vec![1., 7. * PI / 4.], 1.),
                ]
            }
            SphereClassificationProblem::Cube => {
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
        }
    }
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

        let half = SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS);
        assert!((half.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2.],
            vec![vec![PI / 4.]]
        );

        let quarter = SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS);
        assert!((quarter.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_twoquarters_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2., 2f64.sqrt() / 2.],
            vec![vec![PI / 4.], vec![5. * PI / 4.]]
        );

        let two_quarters = SphereClassificationProblem::TwoQuarters(UNIT_CIRCLE_STEPS);
        assert!((two_quarters.evaluate(&Algorithm::ContinuousOneplusoneNA(network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_null_bias() {
        let vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI / 2.
        );

        let half = SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_positive_bias() {
        let vneuron = VNeuron::from_parameters(
            0.5,
            vec![PI / 2.],
            2. * PI / 3.
        );

        let half = SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_negative_bias() {
        let vneuron = VNeuron::from_parameters(
            -0.5,
            vec![3. * PI / 2.],
            PI / 3.
        );

        let half = SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS);
        assert!((half.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_vneuron_oneplusonena_solution() {
        let vneuron = VNeuron::from_parameters(
            2f64.sqrt() / 2.,
            vec![PI / 4.],
            PI / 2.
        );

        let quarter = SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS);
        assert!((quarter.evaluate(&Algorithm::ContinuousBNA(vneuron)) - 1.).abs() < TOL);
    }
}
