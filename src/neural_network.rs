use rand::prelude::*;
use serde_derive::Deserialize;
use rand_distr::Normal;
use std::collections::HashMap;
use cmaes::{DVector, CMAESOptions, Mode, CMAES};
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

#[derive(Debug, Deserialize)]
struct NeuronConfig {
    id: u32,
    inputs: Vec<u32>,
    activation: String,
}

#[derive(Debug, Deserialize)]
pub struct NeuralNetworkConfig {
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    bias_id: Option<u32>,
    neurons: Vec<NeuronConfig>,
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

impl NeuralNetworkConfig {
    pub fn to_neural_network(&self) -> NeuralNetwork {
        let mut neurons = Vec::new();
        for neuron_config in self.neurons.iter() {
            let activation = match neuron_config.activation.as_str() {
                "sigmoid" => SIGMOID,
                "identity" => IDENTITY,
                _ => panic!("Unknown activation function"),
            };

            let mut inputs = Vec::new();
            for input in neuron_config.inputs.iter() {
                inputs.push(NeuronInput::new(*input, None));
            }

            neurons.push(Neuron::new(neuron_config.id, inputs, activation));
        }

        NeuralNetwork {
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            bias_id: self.bias_id,
            neurons,
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
        unimplemented!("Optimization step not implemented for NeuralNetwork");
    }

    fn optimize_with_early_stopping(&mut self, problem: &crate::benchmarks::Benchmark, _max_iters: u32, _fitness_tol: f64, _max_stagnation: Option<u32>) -> u32 where Self: Sized {
        let eval_fn = |x: &DVector<f64>| {
            let network = self.to_network(x);
            problem.evaluate(&Algorithm::NeuralNetwork(network))
        };

        let initial_connection_weights = self.to_vector();

        let mut cmaes_state = CMAESOptions::new(initial_connection_weights, 0.4)
            .mode(Mode::Maximize)
            .build(eval_fn)
            .unwrap();

        let _ = cmaes_state.run();
        let Some(best_individual) = cmaes_state.overall_best_individual() else {
            panic!("No best individual found");
        };
        let generation = cmaes_state.generation() as u32;

        *self = self.to_network(&best_individual.point);
        generation
    }

    fn optimize_cmaes(&mut self, problem: &crate::benchmarks::Benchmark) {
        let eval_fn = |x: &DVector<f64>| {
            let network = self.to_network(x);
            problem.evaluate(&Algorithm::NeuralNetwork(network))
        };

        let initial_connection_weights = self.to_vector();

        let mut cmaes_state = CMAESOptions::new(initial_connection_weights, 0.4)
            .mode(Mode::Maximize)
            .build(eval_fn)
            .unwrap();

        let _ = cmaes_state.run();
        let Some(best_individual) = cmaes_state.overall_best_individual() else {
            panic!("No best individual found");
        };

        *self = self.to_network(&best_individual.point);
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
