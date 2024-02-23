use std::f64::consts::PI;
use rand::prelude::*;
use crate::utils::*;

#[derive(Debug, Clone)]
pub struct VNeuron {
    pub dim: usize,
    pub bias: f64,
    pub angles: Vec<f64>,
    pub bend: f64,
}

impl VNeuron {
    pub fn new(dim: usize) -> Self {
        // genereate random values between 0 and 1
        Self {
            dim,
            bias: random(),
            angles: vec![random(); dim],
            bend: random(),
        }
    }

    pub fn from_params(dim: usize, bias: f64, angles: Vec<f64>, bend: f64) -> Self {
        Self {
            dim,
            bias: (bias + 1.) / 2.,
            angles: angles.iter().map(|x| x / (2. * PI)).collect(),
            bend: bend / (2. * PI),
        }
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> bool {
        let mut normal = vec![0.; self.dim];
        normal[0] = 1.;
        for i in 1..self.dim {
            normal[i] = self.angles[i - 1] * 2. * PI;
        }

        // check that the input is on the right side of the hyperplane
        if polar_dot_product_vect(input, &normal) - (2. * self.bias - 1.) < 0.
        {
            return false;
        }

        // check that the input is in the cone
        for i in 1..self.dim {
            let input_angle = input[i] / (2. * PI);
            if input_angle < self.angles[i-1] - self.bend || input_angle > self.angles[i-1] + self.bend {
                return false;
            }
        }
        true
    }
}
