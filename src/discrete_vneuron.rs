use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use crate::utils::*;
use crate::traits::NeuroevolutionAlgorithm;

#[derive(Debug, Clone)]
pub struct DiscreteVNeuron {
    resolution: usize,
    dim: usize,
    bias: u32,
    angles: Vec<u32>,
    bend: u32,
}

impl fmt::Display for DiscreteVNeuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatted_angles = String::new();
        for angle in &self.angles {
            let formatted_angle = format!("{:.2}", *angle as f64 * 2. * PI / self.resolution as f64);
            formatted_angles.push_str(&formatted_angle);
            formatted_angles.push_str(", ");
        }

        let formatted_bias = format!("{:.2}", 2. * self.bias as f64 / self.resolution as f64 - 1.);
        let formatted_bend = format!("{:.2}", self.bend as f64 * PI / self.resolution as f64);

        write!(
            f,
            "VNeuron {{ dim: {}, bias: {}, angles: [{}], bend: {} }}",
            self.dim, formatted_bias, formatted_angles, formatted_bend
        )
    }
}

impl DiscreteVNeuron {
    pub fn new(dim: usize, resolution: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            resolution,
            dim,
            bias: rng.gen_range(0..=resolution as u32),
            angles: vec![rng.gen_range(0..resolution as u32); dim - 1],
            bend: rng.gen_range(0..resolution as u32),
        }
    }

    fn mutate_component(component: u32, upper_bound: usize) -> u32 {
        let mut rng = thread_rng();
        let sign: i8 = if random::<f64>() < 0.5 { 1 } else { -1 };
        (component as i32 + sign as i32 * sample_harmonic_distribution(&mut rng, upper_bound) as i32 % upper_bound as i32) as u32
    }

    fn get_bias(&self) -> f64 {
        2. * self.bias as f64 / self.resolution as f64 - 1.
    }

    fn get_angle(&self, i: usize) -> f64 {
        self.angles[i] as f64 / self.resolution as f64 * 2. * PI
    }

    fn get_bend(&self) -> f64 {
        self.bend as f64 / self.resolution as f64 * PI
    }
}

impl NeuroevolutionAlgorithm for DiscreteVNeuron {
    fn optimize(&mut self, evaluation_function: fn(&DiscreteVNeuron) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            let mut new_vneuron = self.clone();
            if random::<f64>() < 1. / (self.dim + 1) as f64 {
                new_vneuron.bend = DiscreteVNeuron::mutate_component(self.bend, self.resolution);
            }
            for i in 0..self.dim-1 {
                if random::<f64>() < 1. / (self.dim + 1) as f64 {
                    new_vneuron.angles[i] = DiscreteVNeuron::mutate_component(self.angles[i], self.resolution);
                }
            }
            if random::<f64>() < 1. / (self.dim + 1) as f64 {
                new_vneuron.bias = DiscreteVNeuron::mutate_component(self.bias, self.resolution + 1);
            }

            if evaluation_function(&new_vneuron) > evaluation_function(self) {
                *self = new_vneuron;
            }
        }
    }

    #[allow(unused_variables)]
    fn optimize_cmaes(&mut self, evaluation_function: fn(&Self) -> f64) {
        unimplemented!()
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut normal = vec![0.; self.dim];
        let bias = self.get_bias();
        normal[0] = 1.;
        for i in 1..self.dim {
            normal[i] = self.get_angle(i-1);
        }

        let dot_product = polar_dot_product(input, &normal) - bias;

        if self.get_bias() >= 0. {
            if dot_product < 0. && self.get_bend() < PI / 2. {
                return false;
            }
            if dot_product >= 0. && self.get_bend()>= PI / 2. {
                return true;
            }
        }
        else {
            if dot_product < 0. && self.get_bend() >= PI / 2. {
                return true;
            }
            if dot_product >= 0. && self.get_bend() < PI / 2. {
                return false;
            }
        }

        let norm = (polar_dot_product(input, input) + bias * bias - 2. * bias.abs() * polar_dot_product(input, &normal)).sqrt();
        let cos_angle = dot_product / norm;
        let angle = cos_angle.acos();

        angle <= self.get_bend()
    }
}
