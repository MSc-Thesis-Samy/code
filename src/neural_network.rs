use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashMap;
use cmaes::{DVector, fmax};
use crate::neuroevolution_algorithm::{NeuroevolutionAlgorithm, Algorithm};

pub type ActivationFunction = fn(f64) -> f64;

pub const SIGMOID: ActivationFunction = |x| 1. / (1. + (-4.9 * x).exp());
pub const IDENTITY: ActivationFunction = |x| x;

#[derive(Debug, Clone)]
pub struct NeuronInput {
    input_id: u32,
    weight: f64,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    id: u32,
    inputs: Vec<NeuronInput>,
    activation: ActivationFunction,
}

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    bias_id: Option<u32>,
    neurons: Vec<Neuron>, // Neurons are ordered by layer
}

impl NeuronInput {
    pub fn new(input_id: u32, weight: Option<f64>) -> NeuronInput {
        if let Some(weight) = weight {
            NeuronInput {
                input_id,
                weight,
            }
        } else {
            let weights_distribution = Normal::new(0., 0.8).unwrap();
            NeuronInput {
                input_id,
                weight: weights_distribution.sample(&mut thread_rng()),
            }
        }
    }
}

impl Neuron {
    pub fn new(id: u32, inputs: Vec<NeuronInput>, activation: ActivationFunction) -> Neuron {
        Neuron {
            id,
            inputs,
            activation,
        }
    }
}

impl NeuralNetwork {
    pub fn new(input_ids: Vec<u32>, output_ids: Vec<u32>, bias_id: Option<u32>, neurons: Vec<Neuron>) -> NeuralNetwork {
        NeuralNetwork {
            input_ids,
            output_ids,
            neurons,
            bias_id,
        }
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut values = HashMap::<u32, f64>::new();

        for input_id in self.input_ids.iter() {
            values.insert(*input_id, inputs[*input_id as usize - 1]);
        }

        if let Some(bias_id) = self.bias_id {
            values.insert(bias_id, 1.);
        }

        for neuron in self.neurons.iter() {
            if values.contains_key(&neuron.id) {
                continue;
            }

            if neuron.inputs.is_empty() {
                values.insert(neuron.id, 0.);
                continue;
            }

            let mut sum = 0.;
            for input in neuron.inputs.iter() {
                sum += values.get(&input.input_id).unwrap() * input.weight;
            }
            values.insert(neuron.id, (neuron.activation)(sum));
        }

        let mut outputs = Vec::<f64>::new();
        for output_id in self.output_ids.iter() {
            outputs.push(*values.get(output_id).unwrap());
        }

        outputs
    }

    fn to_vector(&self) -> Vec<f64> {
        let mut connection_weights = Vec::new();
        for neuron in self.neurons.iter() {
            for connection in &neuron.inputs {
                connection_weights.push(connection.weight);
            }
        }

        connection_weights
    }

    fn to_network(&self, connection_weights: &DVector::<f64>) -> NeuralNetwork {
        let mut network = self.clone();
        let mut conn_count = 0;
        for neuron in network.neurons.iter_mut() {
            for connection in &mut neuron.inputs {
                connection.weight = connection_weights[conn_count];
                conn_count += 1;
            }
        }

        network
    }
}

impl NeuroevolutionAlgorithm for NeuralNetwork {
    fn optimization_step(&mut self, _problem: &crate::benchmarks::Benchmark) {
        unimplemented!()
    }

    fn optimize_cmaes(&mut self, problem: &crate::benchmarks::Benchmark) {
        let eval_fn = |x: &DVector<f64>| {
            let network = self.to_network(x);
            problem.evaluate(&Algorithm::NeuralNetworek(network))
        };

        let initial_connection_weights = self.to_vector();
        let solution = fmax(eval_fn, initial_connection_weights, 0.4);
        *self = self.to_network(&solution.point);
    }

    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        let output = self.feed_forward(input);
        output[0]
    }

    fn optimize(&mut self, problem: &crate::benchmarks::Benchmark, _n_iters: u32) {
        self.optimize_cmaes(problem);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward_two_layers() {
        let input_ids = vec![1, 2];
        let output_ids = vec![3];
        let neurons = vec![
            Neuron::new(1, vec![], IDENTITY),
            Neuron::new(2, vec![], IDENTITY),
            Neuron::new(3, vec![NeuronInput::new(1, Some(0.5)), NeuronInput::new(2, Some(0.5))], IDENTITY),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, None, neurons);

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![1.]);
    }

    #[test]
    fn test_feed_forward_three_layers() {
        let input_ids = vec![1, 2];
        let output_ids = vec![4];
        let neurons = vec![
            Neuron::new(1, vec![] , IDENTITY),
            Neuron::new(2, vec![], IDENTITY),
            Neuron::new(3, vec![NeuronInput::new(1, Some(0.5)), NeuronInput::new(2, Some(0.5))], IDENTITY),
            Neuron::new(4, vec![NeuronInput::new(3, Some(0.5))], IDENTITY),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, None, neurons);

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![0.5]);
    }

    #[test]
    fn test_feed_forward_two_layers_with_bias() {
        let input_ids = vec![1, 2];
        let output_ids = vec![3];
        let neurons = vec![
            Neuron::new(1, vec![], IDENTITY),
            Neuron::new(2, vec![], IDENTITY),
            Neuron::new(3, vec![NeuronInput::new(1, Some(0.5)), NeuronInput::new(2, Some(0.5)), NeuronInput::new(4, Some(1.))], IDENTITY),
            Neuron::new(4, vec![], IDENTITY),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, Some(4), neurons);

        let inputs = vec![2., 2.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![3.]);
    }
}
