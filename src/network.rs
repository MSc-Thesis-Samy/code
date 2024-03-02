use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::Exp;
use crate::utils::*;
use crate::traits::NeuroevolutionAlgorithm;

#[derive(Debug, Clone)]
pub struct Network {
    n_neurons: usize,
    dim: usize,
    biases: Vec<f64>,
    angles: Vec<Vec<f64>>,
    output_layer: fn(&Vec<bool>) -> bool,
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let formatted_biases: Vec<String> = self.biases.iter()
            .map(|&bias| format!("{:.2}", 2. * bias - 1.))
            .collect();

        let formatted_angles: Vec<Vec<String>> = self.angles.iter()
            .map(|row| {
                row.iter()
                    .map(|&angle| format!("{:.2}", angle * 2. * PI))
                    .collect()
            })
            .collect();

        writeln!(f, "Number of Neurons: {}", self.n_neurons)?;
        writeln!(f, "Dimension: {}", self.dim)?;
        writeln!(f, "Biases: {:?}", formatted_biases)?;
        writeln!(f, "Angles: {:?}", formatted_angles)?;

        Ok(())
    }
}

impl Network {
    pub fn new(n_neurons: usize, dim: usize) -> Self {
        Self {
            n_neurons,
            dim,
            biases: vec![random(); n_neurons],
            angles: vec![vec![random(); dim - 1]; n_neurons],
            output_layer: |inputs: &Vec<bool>| inputs.iter().any(|&x| x),
        }
    }

    fn mutate_component(component: f64) -> f64 {
        let exp = Exp::new(1.).unwrap();
        let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
        let mut new_component = component + sign * exp.sample(&mut thread_rng());
        new_component -= new_component.floor();
        new_component
    }

    // pub fn get_network(x: &DVector<f64>) -> Network {
    //     let x: DVector<f64> = x.map(|x| x - x.floor());
    //     let mut angles = [[0.;D];N];
    //     for i in 0..N {
    //         for j in 0..D {
    //             angles[i][j] = x[i * D + j];
    //         }
    //     }
    //     Network {
    //         angles,
    //         output_layer: |hidden: &[bool; N]| hidden.iter().any(|&x| x),
    //     }
    // }
}

impl NeuroevolutionAlgorithm for Network {
    fn optimize(&mut self, evaluation_function: fn(&Network) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            let mut new_network = self.clone();
            for i in 0..self.n_neurons {
                new_network.biases[i] = Network::mutate_component(self.biases[i]);
                for j in 0..self.dim-1 {
                    if random::<f64>() < 1. / (self.dim as f64 * self.n_neurons as f64) {
                        new_network.angles[i][j] = Network::mutate_component(self.angles[i][j]);
                    }
                }
            }

            if evaluation_function(&new_network) > evaluation_function(self) {
                *self = new_network;
            }
        }
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut hidden = vec![false; self.dim];
        for i in 0..self.n_neurons {
            let mut normal = vec![0.;self.dim];
            normal[0] = 1.;
            for j in 1..self.dim {
                normal[j] = self.angles[i][j-1] * 2. * PI;
            }
            hidden[i] = polar_dot_product(input, &normal) - (2. * self.biases[i] - 1.) > 0.;
        }
        (self.output_layer)(&hidden)
    }
}
