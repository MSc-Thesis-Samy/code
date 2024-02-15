use std::fmt;
use rand::prelude::*;
use rand_distr::{Exp, Distribution};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Network<const N: usize, const D: usize> {
    parameters: [[f64;D];N],
    output_layer: fn(&[bool; N]) -> bool,
}

fn generate_random_array<const N: usize, const D: usize>() -> [[f64;D];N] {
    let mut array = [[0.;D];N];
    for i in 0..N {
        for j in 0..D {
            array[i][j] = random::<f64>();
        }
    }
    array
}

fn polar_dot_product<const D: usize>(v1: &[f64; D], v2: &[f64; D]) -> f64 {
    let (r1, angles1) = v1.split_first().unwrap();
    let (r2, angles2) = v2.split_first().unwrap();

    let norm_product = r1 * r2;
    let angle_difference_cosine: f64 = angles1
        .iter()
        .zip(angles2.iter())
        .map(|(&theta1, &theta2)| (theta1 - theta2).cos())
        .product();

    norm_product * angle_difference_cosine
}

impl<const N: usize, const D: usize> std::fmt::Display for Network<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, param) in self.parameters.iter().enumerate() {
            write!(f, "Neuron {}: [", index)?;
            for (i, &value) in param.iter().enumerate() {
                if i == 0 {
                    write!(f, "{:.2}", 2. * value - 1.)?;
                } else {
                    write!(f, "{:.2}", value * 2. * PI)?;
                }
                if i != param.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]\n")?;
        }
        Ok(())
    }
}

impl<const N: usize, const D: usize> Network<N, D> {
    pub fn new() -> Self {
        Self {
            parameters: generate_random_array::<N, D>(),
            output_layer: |inputs: &[bool; N]| inputs.iter().any(|&x| x), // OR
        }
    }

    pub fn optimize(&mut self, evaluation_function: fn(&Network<N, D>) -> f64, n_iters: u32) {
        let exp = Exp::new(1.).unwrap();

        for _ in 0..n_iters {
            let mut new_network = self.clone();
            for i in 0..N {
                for j in 0..D {
                    if random::<f64>() < 1. / (2. * N as f64) {
                        let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
                        new_network.parameters[i][j] += sign * exp.sample(&mut thread_rng());
                        new_network.parameters[i][j] -= new_network.parameters[i][j].floor();
                    }
                }
            }

            if evaluation_function(&new_network) > evaluation_function(self) {
                *self = new_network;
            }
        }
    }

    pub fn evaluate(&self, input: &[f64; D]) -> bool {
        let mut hidden = [false; N];
        for i in 0..N {
            let mut normal = [0.;D];
            normal[0] = 1.;
            for j in 1..D {
                normal[j] = self.parameters[i][j] * 2. * PI;
            }
            hidden[i] = polar_dot_product(input, &normal) - (2. * self.parameters[i][0] - 1.) > 0.;
        }
        (self.output_layer)(&hidden)
    }
}
