use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::{Exp, Distribution};
use cmaes::DVector;
use crate::utils::*;
use crate::traits::NetworkTrait;

#[derive(Debug, Clone)]
pub struct Network<const N: usize, const D: usize> {
    parameters: [[f64;D];N],
    output_layer: fn(&[bool; N]) -> bool,
}

impl<const N: usize, const D: usize> fmt::Display for Network<N, D> {
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

impl<const N: usize, const D: usize> NetworkTrait<N, D> for Network<N, D> {
    fn optimize(&mut self, evaluation_function: fn(&Network<N, D>) -> f64, n_iters: u32) {
        let exp = Exp::new(1.).unwrap();

        for _ in 0..n_iters {
            let mut new_network = self.clone();
            for i in 0..N {
                for j in 0..D {
                    if random::<f64>() < 1. / (D as f64 * N as f64) {
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

    fn evaluate(&self, input: &[f64; D]) -> bool {
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

impl<const N: usize, const D: usize> Network<N, D> {
    pub fn new() -> Self {
        Self {
            parameters: generate_random_float_array::<N, D>(),
            output_layer: |inputs: &[bool; N]| inputs.iter().any(|&x| x), // OR
        }
    }

    pub fn get_network(x: &DVector<f64>) -> Network<N, D> {
        let x: DVector<f64> = x.map(|x| x - x.floor());
        let mut parameters = [[0.;D];N];
        for i in 0..N {
            for j in 0..D {
                parameters[i][j] = x[i * D + j];
            }
        }
        Network {
            parameters,
            output_layer: |hidden: &[bool; N]| hidden.iter().any(|&x| x),
        }
    }
}
