use std::f64::consts::PI;
use crate::traits::NeuroevolutionAlgorithm;

const UNIT_CIRCLE_STEPS: u32 = 100;

pub fn full<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm,
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

pub fn half<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm,
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

pub fn quarter<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm,
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

pub fn two_quarters<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm,
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

pub fn square<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm,
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

pub fn cube<N>(alg: &N) -> f64
where
    N: NeuroevolutionAlgorithm
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

    const TOL: f64 = 5e-2;

    #[test]
    fn test_half_network() {
        let network = Network::from_parameters(
            vec![0.],
            vec![vec![PI / 2.]]
        );

        assert!((half(&network) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2.],
            vec![vec![PI / 4.]]
        );

        assert!((quarter(&network) - 1.).abs() < TOL);
    }

    #[test]
    fn test_twoquarters_network() {
        let network = Network::from_parameters(
            vec![2f64.sqrt() / 2., 2f64.sqrt() / 2.],
            vec![vec![PI / 4.], vec![5. * PI / 4.]]
        );

        assert!((two_quarters(&network) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_null_bias() {
        let vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI / 2.
        );

        assert!((half(&vneuron) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_positive_bias() {
        let vneuron = VNeuron::from_parameters(
            0.5,
            vec![PI / 2.],
            2. * PI / 3.
        );

        assert!((half(&vneuron) - 1.).abs() < TOL);
    }

    #[test]
    fn test_half_vneuron_negative_bias() {
        let vneuron = VNeuron::from_parameters(
            -0.5,
            vec![3. * PI / 2.],
            PI / 3.
        );

        assert!((half(&vneuron) - 1.).abs() < TOL);
    }

    #[test]
    fn test_quarter_vneuron_oneplusonena_solution() {
        let vneuron = VNeuron::from_parameters(
            2f64.sqrt() / 2.,
            vec![PI / 4.],
            PI / 2.
        );

        assert!((quarter(&vneuron) - 1.).abs() < TOL);
    }

    #[test]
    fn test_full_vneuron_null_bias() {
        let vneuron = VNeuron::from_parameters(
            0.,
            vec![PI / 2.],
            PI
        );

        assert!((full(&vneuron) - 1.).abs() < TOL);
    }

    #[test]
    fn test_full_vneuron_negative_bias() {
        let vneuron = VNeuron::from_parameters(
            -1.,
            vec![PI / 2.],
            PI / 2.
        );

        assert!((full(&vneuron) - 1.).abs() < TOL);
    }
}
