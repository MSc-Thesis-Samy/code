use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::{Exp, Distribution};
use crate::utils::*;
use crate::neuroevolution_algorithm::*;

#[derive(Debug, Clone)]
pub struct VNeuron {
    dim: usize,
    bias: f64,
    angles: Vec<f64>,
    bend: f64,
}

impl fmt::Display for VNeuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatted_angles = String::new();
        for angle in &self.angles {
            let formatted_angle = format!("{:.2}", angle * 2.0 * PI);
            formatted_angles.push_str(&formatted_angle);
            formatted_angles.push_str(", ");
        }

        let formatted_bias = format!("{:.2}", self.bias * 2.0 - 1.0);
        let formatted_bend = format!("{:.2}", self.bend * PI);

        write!(
            f,
            "VNeuron {{ dim: {}, bias: {}, angles: [{}], bend: {} }}",
            self.dim, formatted_bias, formatted_angles, formatted_bend
        )
    }
}

impl VNeuron {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            bias: random(),
            angles: vec![random(); dim - 1],
            bend: random(),
        }
    }

    pub fn from_parameters(bias: f64, angles: Vec<f64>, bend: f64) -> Self {
        Self {
            dim: angles.len() + 1,
            bias: (bias + 1.) / 2.,
            angles: angles.iter().map(|x| x / (2. * PI)).collect(),
            bend: bend / PI,
        }
    }

    fn mutate_component(component: f64) -> f64 {
        let exp = Exp::new(1.).unwrap();
        let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
        let mut new_component = component + sign * exp.sample(&mut thread_rng());
        new_component -= new_component.floor();
        new_component
    }

    fn get_bias(&self) -> f64 {
        2. * self.bias - 1.
    }

    fn get_angle(&self, i: usize) -> f64 {
        self.angles[i] * 2. * PI
    }

    fn get_bend(&self) -> f64 {
        self.bend * PI
    }
}

impl NeuroevolutionAlgorithm for VNeuron {
    fn optimize(&mut self, evaluation_function: fn(&Algorithm) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            let mut new_vneuron = self.clone();
            if random::<f64>() < 1. / (self.dim + 1) as f64  {
                new_vneuron.bend = VNeuron::mutate_component(new_vneuron.bend);
            }
            for i in 0..self.dim-1 {
                if random::<f64>() < 1. / (self.dim + 1) as f64 {
                    new_vneuron.angles[i] = VNeuron::mutate_component(new_vneuron.angles[i]);
                }
            }
            if random::<f64>() < 1. / (self.dim + 1) as f64 {
                new_vneuron.bias = VNeuron::mutate_component(new_vneuron.bias);
            }

            if evaluation_function(&Algorithm::ContinuousBNA(new_vneuron.clone())) > evaluation_function(&Algorithm::ContinuousBNA(self.clone())) {
                *self = new_vneuron;
            }
        }
    }

    #[allow(unused_variables)]
    fn optimize_cmaes(&mut self, evaluation_function: fn(&Algorithm) -> f64) {
        unimplemented!()
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut normal = vec![0.; self.dim];
        let bias = self.get_bias();
        normal[0] = 1.;
        for i in 1..self.dim {
            normal[i] = self.get_angle(i-1);
        }

        let dot_product = polar_dot_product(input, &normal) - bias.abs();
        let norm = (polar_dot_product(input, input) + bias * bias - 2. * bias.abs() * polar_dot_product(input, &normal)).sqrt();
        let cos_angle = dot_product / norm;
        let angle = cos_angle.acos();

        if bias >= 0. {
            angle <= self.get_bend()
        }
        else {
            PI - angle <= self.get_bend()
        }
    }
}
