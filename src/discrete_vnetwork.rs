use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use crate::utils::*;
use crate::neuroevolution_algorithm::*;
use crate::benchmarks::Benchmark;

#[derive(Debug, Clone)]
pub struct DiscreteVNetwork {
    n_neurons: usize,
    dim: usize,
    resolution: usize,
    biases: Vec<u32>,
    bends: Vec<u32>,
    angles: Vec<Vec<u32>>,
    output_layer: fn(&Vec<bool>) -> bool,
}

impl fmt::Display for DiscreteVNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let formatted_biases: Vec<String> = self.biases.iter()
            .map(|&bias| format!("{:.2}", 2. * bias as f64 / self.resolution as f64 - 1.))
            .collect();

        let formatted_bends: Vec<String> = self.bends.iter()
            .map(|&bend| format!("{:.2}", bend as f64 * PI / self.resolution as f64))
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
        writeln!(f, "Bends: {:?}", formatted_bends)?;

        Ok(())
    }
}

impl DiscreteVNetwork {
    pub fn new(resolution: usize, n_neurons: usize, dim: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            resolution,
            n_neurons,
            dim,
            biases: vec![rng.gen_range(0..=resolution as u32); n_neurons],
            angles: vec![vec![rng.gen_range(0..resolution as u32); dim - 1]; n_neurons],
            bends: vec![rng.gen_range(0..resolution as u32); n_neurons],
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

    fn get_bend(&self, i: usize) -> f64 {
        self.bends[i] as f64 / self.resolution as f64 * PI
    }

    pub fn get_biases(&self) -> Vec<f64> {
        self.biases.iter().map(|&x| 2. * x as f64 / self.resolution as f64 - 1.).collect()
    }

    pub fn get_angles(&self) -> Vec<Vec<f64>> {
        self.angles.iter().map(|row| row.iter().map(|&x| x as f64 / self.resolution as f64 * 2. * PI).collect()).collect()
    }

    pub fn get_bends(&self) -> Vec<f64> {
        self.bends.iter().map(|&x| x as f64 / self.resolution as f64 * PI).collect()
    }

    fn evaluate_core(&self, input: &Vec<f64>) -> bool {
        let mut hidden = vec![false; self.n_neurons];
        for i in 0..self.n_neurons {
            let mut normal = vec![0.; self.dim];
            let bias = self.get_bias(i);
            normal[0] = 1.;
            for j in 1..self.dim {
                normal[j] = self.get_angle(i, j-1);
            }

            let dot_product = polar_dot_product(input, &normal) - bias.abs();
            let norm = (polar_dot_product(input, input) + bias * bias - 2. * bias.abs() * polar_dot_product(input, &normal)).sqrt();
            let cos_angle = dot_product / norm;
            let angle = cos_angle.acos();

            if bias >= 0. {
                hidden[i] = angle <= self.get_bend(i)
            }
            else {
                hidden[i] = PI - angle <= self.get_bend(i)
            }
        }

        (self.output_layer)(&hidden)
    }
}

impl NeuroevolutionAlgorithm for DiscreteVNetwork {
    fn optimization_step(&mut self, problem: &Benchmark) {
        let mut new_network = self.clone();
        for i in 0..self.n_neurons {
            if random::<f64>() < 1. / ((self.dim + 1) * self.n_neurons) as f64 {
                new_network.biases[i] = DiscreteVNetwork::mutate_component(self.biases[i], self.resolution + 1);
            }

            if random::<f64>() < 1. / ((self.dim + 1) * self.n_neurons) as f64 {
                new_network.bends[i] = DiscreteVNetwork::mutate_component(self.bends[i], self.resolution);
            }

            for j in 0..self.dim-1 {
                if random::<f64>() < 1. / ((self.dim + 1) * self.n_neurons) as f64 {
                    new_network.angles[i][j] = DiscreteVNetwork::mutate_component(self.angles[i][j], self.resolution);
                }
            }
        }

        if problem.evaluate(&Algorithm::DiscreteBNA(new_network.clone())) > problem.evaluate(&Algorithm::DiscreteBNA(self.clone())) {
            *self = new_network;
        }
    }

    #[allow(unused_variables)]
    fn optimize_cmaes(&mut self, problem: &Benchmark) {
        unimplemented!()
    }

    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        if self.evaluate_core(input) {
            1.
        } else {
            0.
        }
    }
}
