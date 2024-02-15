use std::fmt;
use rand::prelude::*;
use rand_distr::{Exp, Distribution};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Network<const N: usize> {
    angles: [f64; N],
    biases: [f64; N],
    output_layer: fn(&[bool; N]) -> bool,
}

fn generate_random_array<const N: usize>() -> [f64; N] {
    let mut array = [0.; N];
    for i in 0..N {
        array[i] = random::<f64>();
    }
    array
}

fn polar_dot_product((r1, theta1): (f64, f64), (r2, theta2): (f64, f64)) -> f64 {
    r1 * r2 * (theta1 - theta2).cos()
}

impl<const N: usize> std::fmt::Display for Network<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Network {{ angles: [{}], biases: [{}] }}",
            self.angles
                .iter()
                .map(|&x| format!("{:.2}", 2. * PI * x))
                .collect::<Vec<_>>()
                .join(", "),
            self.biases
                .iter()
                .map(|&x| format!("{:.2}", 2. * x - 1.))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl<const N: usize> Network<N> {
    pub fn new() -> Self {
        Self {
            angles: generate_random_array(),
            biases: generate_random_array(),
            output_layer: |inputs: &[bool; N]| inputs.iter().any(|&x| x), // OR
        }
    }

    pub fn optimize(&mut self, evaluation_function: fn(&Network<N>) -> f64, n_iters: u32) {
        let exp = Exp::new(1.).unwrap();

        for _ in 0..n_iters {
            let mut new_network = self.clone();
            for i in 0..N {
                if random::<f64>() < 1. / (2. * N as f64) {
                    let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
                    new_network.angles[i] += sign * exp.sample(&mut thread_rng());
                    new_network.biases[i] += sign * exp.sample(&mut thread_rng());
                    new_network.angles[i] -= new_network.angles[i].floor();
                    new_network.biases[i] -= new_network.biases[i].floor();
                }
            }

            if evaluation_function(&new_network) > evaluation_function(self) {
                *self = new_network;
            }
        }
    }

    pub fn evaluate(&self, input: (f64, f64)) -> bool {
        let mut hidden = [false; N];
        for i in 0..N {
            let normal = (1., self.angles[i] * 2. * PI);
            hidden[i] = polar_dot_product(input, normal) - (2. * self.biases[i] - 1.) > 0.;
        }
        (self.output_layer)(&hidden)
    }
}
