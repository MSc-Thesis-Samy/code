use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use crate::benchmarks::ClassificationProblemEval;
use crate::utils::*;
use crate::neuroevolution_algorithm::*;
use crate::benchmarks::ClassificationProblem;

#[derive(Debug, Clone)]
pub struct DiscreteNetwork {
    n_neurons: usize,
    dim: usize,
    resolution: usize,
    biases: Vec<u32>,
    angles: Vec<Vec<u32>>,
    output_layer: fn(&Vec<bool>) -> bool,
}

impl fmt::Display for DiscreteNetwork {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let formatted_biases: Vec<String> = self.biases.iter()
            .map(|&bias| format!("{:.2}", 2. * bias as f64 / self.resolution as f64 - 1.))
            .collect();

        let formatted_angles: Vec<Vec<String>> = self.angles.iter()
            .map(|row| {
                row.iter()
                    .map(|&angle| format!("{:.2}", angle as f64 / self.resolution as f64 * 2. * PI))
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

impl DiscreteNetwork {
    pub fn new(resolution: usize, n_neurons: usize, dim: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            resolution,
            n_neurons,
            dim,
            biases: vec![rng.gen_range(0..=resolution as u32); n_neurons],
            angles: vec![vec![rng.gen_range(0..resolution as u32); dim - 1]; n_neurons],
            output_layer: |inputs: &Vec<bool>| inputs.iter().any(|&x| x),
        }
    }

    fn mutate_component(component: u32, upper_bound: usize) -> u32 {
        let mut rng = thread_rng();
        let sign: i8 = if random::<f64>() < 0.5 { 1 } else { -1 };
        ((component as i32 + sign as i32 * sample_harmonic_distribution(&mut rng, upper_bound) as i32).rem_euclid(upper_bound as i32)) as u32
    }

    fn get_bias(&self, i: usize) -> f64 {
        2. * self.biases[i] as f64 / self.resolution as f64 - 1.
    }

    fn get_angle(&self, i: usize, j: usize) -> f64 {
        self.angles[i][j] as f64 / self.resolution as f64 * 2. * PI
    }

    pub fn get_biases(&self) -> Vec<f64> {
        self.biases.iter().map(|&x| 2. * x as f64 / self.resolution as f64 - 1.).collect()
    }

    pub fn get_angles(&self) -> Vec<Vec<f64>> {
        self.angles.iter().map(|row| row.iter().map(|&x| x as f64 / self.resolution as f64 * 2. * PI).collect()).collect()
    }
}

impl NeuroevolutionAlgorithm for DiscreteNetwork {
    fn optimization_step(&mut self, problem: &ClassificationProblem) {
        let mut new_network = self.clone();
        for i in 0..self.n_neurons {
            new_network.biases[i] = DiscreteNetwork::mutate_component(self.biases[i], self.resolution + 1);
            for j in 0..self.dim-1 {
                new_network.angles[i][j] = DiscreteNetwork::mutate_component(self.angles[i][j], self.resolution);
            }
        }

        if problem.evaluate(&Algorithm::DiscreteOneplusoneNA(new_network.clone())) > problem.evaluate(&Algorithm::DiscreteOneplusoneNA(self.clone())) {
            *self = new_network;
        }
    }

    #[allow(unused_variables)]
    fn optimize_cmaes(&mut self, problem: &ClassificationProblem) {
        unimplemented!()
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut hidden = vec![false; self.n_neurons];
        for i in 0..self.n_neurons {
            let mut normal = vec![0.;self.dim];
            normal[0] = 1.;
            for j in 1..self.dim {
                normal[j] = self.get_angle(i, j-1);
            }
            if 2. * self.biases[i] as f64 / self.resolution as f64 - 1. >= 0. {
                hidden[i] = polar_dot_product(input, &normal) - self.get_bias(i).abs() >= 0.;
            }
            else {
                hidden[i] = polar_dot_product(input, &normal) - self.get_bias(i).abs() < 0.;
            }
        }
        (self.output_layer)(&hidden)
    }
}
