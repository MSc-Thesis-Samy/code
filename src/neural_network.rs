#![allow(dead_code)] // TODO remove
//
use std::collections::HashMap;

pub struct NeuronInput {
    input_id: u32,
    weight: f32,
}

pub struct Neuron {
    id: u32,
    inputs: Vec<NeuronInput>,
}

pub struct NeuralNetwork {
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    neurons: Vec<Neuron>, // Neurons are ordered by layer
}

impl NeuronInput {
    pub fn new(input_id: u32, weight: f32) -> NeuronInput {
        NeuronInput {
            input_id,
            weight,
        }
    }
}

impl Neuron {
    pub fn new(id: u32, inputs: Vec<NeuronInput>) -> Neuron {
        Neuron {
            id,
            inputs,
        }
    }
}

impl NeuralNetwork {
    pub fn new(input_ids: Vec<u32>, output_ids: Vec<u32>, neurons: Vec<Neuron>) -> NeuralNetwork {
        NeuralNetwork {
            input_ids,
            output_ids,
            neurons,
        }
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut values = HashMap::<u32, f32>::new();

        for input_id in self.input_ids.iter() {
            values.insert(*input_id, inputs[*input_id as usize - 1]);
        }

        for neuron in self.neurons.iter() {
            if values.contains_key(&neuron.id) {
                continue;
            }

            let mut sum = 0.;
            for input in neuron.inputs.iter() {
                sum += values.get(&input.input_id).unwrap() * input.weight;
            }
            values.insert(neuron.id, sum);
        }

        let mut outputs = Vec::<f32>::new();
        for output_id in self.output_ids.iter() {
            outputs.push(*values.get(output_id).unwrap());
        }

        outputs
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
            Neuron::new(1, vec![]),
            Neuron::new(2, vec![]),
            Neuron::new(3, vec![NeuronInput::new(1, 0.5), NeuronInput::new(2, 0.5)]),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, neurons);

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(inputs);

        assert_eq!(outputs, vec![1.]);
    }

    #[test]
    fn test_feed_forward_three_layers() {
        let input_ids = vec![1, 2];
        let output_ids = vec![4];
        let neurons = vec![
            Neuron::new(1, vec![]),
            Neuron::new(2, vec![]),
            Neuron::new(3, vec![NeuronInput::new(1, 0.5), NeuronInput::new(2, 0.5)]),
            Neuron::new(4, vec![NeuronInput::new(3, 0.5)]),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, neurons);

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(inputs);

        assert_eq!(outputs, vec![0.5]);
    }
}
