use std::f64::consts::PI;
use crate::neuroevolution_algorithm::*;

const UNIT_CIRCLE_STEPS: u32 = 100;

pub fn full(alg: &Algorithm) -> f64
{
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = alg.evaluate(&vec![1., angle]);
        if output {
            sum += 1;
        }
    }

    sum as f64 / UNIT_CIRCLE_STEPS as f64
}

pub fn half(alg: &Algorithm) -> f64
{
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = alg.evaluate(&vec![1., angle]);
        if output && angle <= PI || !output && angle > PI {
            sum += 1;
        }
    }

    sum as f64 / UNIT_CIRCLE_STEPS as f64
}

pub fn quarter(alg: &Algorithm) -> f64
{
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = alg.evaluate(&vec![1., angle]);
        if output && angle <= PI / 2. || !output && angle > PI / 2. {
            sum += 1;
        }
    }

    sum as f64 / UNIT_CIRCLE_STEPS as f64
}

pub fn two_quarters(alg: &Algorithm) -> f64
{
    let mut sum = 0;
    for i in 0..UNIT_CIRCLE_STEPS {
        let angle = 2. * PI * i as f64 / UNIT_CIRCLE_STEPS as f64;
        let output = alg.evaluate(&vec![1., angle]);
        if output && (angle <= PI / 2. || angle >= PI && angle <= 3. * PI / 2.)
        || !output && (angle > PI / 2. && angle < PI || angle > 3. * PI / 2.) {
            sum += 1;
        }
    }

    sum as f64 / UNIT_CIRCLE_STEPS as f64
}

pub fn square(alg: &Algorithm) -> f64
{
    let points_with_labels = [
        (1., PI / 4., true),
        (1., 3. * PI / 4., false),
        (1., 5. * PI / 4., true),
        (1., 7. * PI / 4., false),
    ];

    points_with_labels
        .iter()
        .map(|&(r, theta, label)| {
            let output = alg.evaluate(&vec![r, theta]);
            if output && label || !output && !label {
                1.
            } else {
                0.
            }
        })
        .sum::<f64>() / 4.
}

pub fn cube(alg: &Algorithm) -> f64
{
    let points_with_labels = [
        (1., PI / 4., PI / 4., true),
        (1., 3. * PI / 4., PI / 4., false),
        (1., 5. * PI / 4., PI / 4., true),
        (1., 7. * PI / 4., PI / 4., false),
        (1., PI / 4., 3. * PI / 4., true),
        (1., 3. * PI / 4., 3. * PI / 4., false),
        (1., 5. * PI / 4., 3. * PI / 4., true),
        (1., 7. * PI / 4., 3. * PI / 4., false),
    ];

    points_with_labels
        .iter()
        .map(|&(r, theta, phi, label)| {
            let output = alg.evaluate(&vec![r, theta, phi]);
            if output && label || !output && !label {
                1.
            } else {
                0.
            }
        })
        .sum::<f64>() / 8.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::vneuron::VNeuron;
    use crate::neuroevolution_algorithm::Algorithm;

    const TOL: f64 = 5e-2;

    #[test]
    fn test_half_network() {
        let mut network = Network::from_parameters(
            vec![0.],
            vec![vec![PI / 2.]]
        );

        assert!((half(&Algorithm::ContinuousOneplusoneNA(&mut network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_network() {
        let mut network = Network::from_parameters(
            vec![2f64.sqrt() / 2.],
            vec![vec![PI / 4.]]
        );

        assert!((quarter(&Algorithm::ContinuousOneplusoneNA(&mut network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_twoquarters_network() {
        let mut network = Network::from_parameters(
            vec![2f64.sqrt() / 2., 2f64.sqrt() / 2.],
            vec![vec![PI / 4.], vec![5. * PI / 4.]]
        );

        assert!((two_quarters(&Algorithm::ContinuousOneplusoneNA(&mut network)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_null_bias() {
        let mut vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI / 2.
        );

        assert!((half(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_positive_bias() {
        let mut vneuron = VNeuron::from_parameters(
            0.5,
            vec![PI / 2.],
            2. * PI / 3.
        );

        assert!((half(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_negative_bias() {
        let mut vneuron = VNeuron::from_parameters(
            -0.5,
            vec![3. * PI / 2.],
            PI / 3.
        );

        assert!((half(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_vneuron_oneplusonena_solution() {
        let mut vneuron = VNeuron::from_parameters(
            2f64.sqrt() / 2.,
            vec![PI / 4.],
            PI / 2.
        );

        assert!((quarter(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_full_vneuron_null_bias() {
        let mut vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI
        );

        assert!((full(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }

    #[test]
    fn test_full_vneuron_negative_bias() {
        let mut vneuron = VNeuron::from_parameters(
            -1.,
            vec![PI / 2.],
            PI / 2.
        );

        assert!((full(&Algorithm::ContinuousBNA(&mut vneuron)) - 1.).abs() < TOL);
    }
}
