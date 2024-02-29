use std::fmt;
use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::{Exp, Distribution};
use crate::utils::*;
use crate::traits::VNeuronTrait;

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

    pub fn from_params(dim: usize, bias: f64, angles: Vec<f64>, bend: f64) -> Self {
        Self {
            dim,
            bias: (bias + 1.) / 2.,
            angles: angles.iter().map(|x| x / (2. * PI)).collect(),
            bend: bend / PI,
        }
    }

    fn mutate_component(component: &mut f64) {
        let exp = Exp::new(1.).unwrap();
        let sign = if random::<f64>() < 0.5 { 1. } else { -1. };
        *component += sign * exp.sample(&mut thread_rng());
        *component -= component.floor();
    }
}

impl VNeuronTrait for VNeuron {
    fn optimize(&mut self, evaluation_function: fn(&VNeuron) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            let mut new_vneuron = self.clone();
            if random::<f64>() < 1. / 3. {
                VNeuron::mutate_component(&mut new_vneuron.bend);
            }
            for i in 0..self.dim-1 {
                if random::<f64>() < 1. / 3. {
                    VNeuron::mutate_component(&mut new_vneuron.angles[i]);
                }
            }
            if random::<f64>() < 1. / 3. {
                VNeuron::mutate_component(&mut new_vneuron.bias);
            }

            if evaluation_function(&new_vneuron) > evaluation_function(self) {
                *self = new_vneuron;
            }
        }
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut normal = vec![0.; self.dim];
        let bias = 2. * self.bias - 1.;
        normal[0] = 1.;
        for i in 1..self.dim {
            normal[i] = self.angles[i - 1] * 2. * PI;
        }

        let dot_product = polar_dot_product_vect(input, &normal) - bias;

        if dot_product < 0. && self.bend * PI < PI / 2. {
            return false;
        }
        if dot_product >= 0. && self.bend * PI >= PI / 2. {
            return true;
        }

        let norm = (polar_dot_product_vect(input, input) + bias * bias - 2. * bias * polar_dot_product_vect(input, &normal)).sqrt();
        let cos_angle = dot_product / norm;
        let angle = cos_angle.acos();

        angle <= self.bend * PI
    }
}
