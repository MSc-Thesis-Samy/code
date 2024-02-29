use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use crate::utils::*;
use crate::traits::NetworkTrait;

#[derive(Debug, Clone)]
pub struct DiscreteNetwork<const N: usize, const D: usize> {
    resolution: usize,
    parameters: [[u32;D];N],
    output_layer: fn(&[bool; N]) -> bool,
}

impl<const N: usize, const D: usize> fmt::Display for DiscreteNetwork<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, param) in self.parameters.iter().enumerate() {
            write!(f, "Neuron {}: [", index)?;
            for (i, &value) in param.iter().enumerate() {
                if i == 0 {
                    let bias = 2. * value as f64 / self.resolution as f64 - 1.;
                    write!(f, "{:.2}", bias)?;
                } else {
                    let angle = 2. * PI * value as f64 / self.resolution as f64;
                    write!(f, "{:.2}", angle)?;
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

impl<const N: usize, const D: usize> DiscreteNetwork<N, D> {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            parameters: generate_random_int_array::<N, D>(resolution),
            output_layer: |inputs: &[bool; N]| inputs.iter().any(|&x| x), // OR
        }
    }
}

impl<const N: usize, const D: usize> NetworkTrait<N, D> for DiscreteNetwork<N, D> {
    fn optimize(&mut self, evaluation_function: fn(&DiscreteNetwork<N, D>) -> f64, n_iters: u32) {
        let mut rng = thread_rng();

        for _ in 0..n_iters {
            let mut new_network = self.clone();
            for i in 0..N {
                if random::<f64>() < 1. / (D as f64 * N as f64) {
                    let sign: i8 = if random::<f64>() < 0.5 { 1 } else { -1 };
                    new_network.parameters[i][0] = ((new_network.parameters[i][0] as i32 + sign as i32 * sample_harmonic_distribution(&mut rng, self.resolution) as i32) % self.resolution as i32 + 1) as u32;
                }
                for j in 1..D {
                    if random::<f64>() < 1. / (D as f64 * N as f64) {
                        let sign: i8 = if random::<f64>() < 0.5 { 1 } else { -1 };
                        new_network.parameters[i][j] = ((new_network.parameters[i][j] as i32 + sign as i32 * sample_harmonic_distribution(&mut rng, self.resolution) as i32) % self.resolution as i32) as u32;
                    }
                }
            }

            if evaluation_function(&new_network) > evaluation_function(self) {
                *self = new_network;
            }
        }
    }

    fn evaluate(&self, input: &[f64; D]) -> bool {
        let mut hidden = [false; N];
        for i in 0..N {
            let mut normal = [0.;D];
            normal[0] = 1.;
            for j in 1..D {
                normal[j] = self.parameters[i][j] as f64 / self.resolution as f64 * 2. * PI;
            }
            hidden[i] = polar_dot_product(input, &normal) - (2. * self.parameters[i][0] as f64 / self.resolution as f64 - 1.) > 0.;
        }
        (self.output_layer)(&hidden)
    }
}
