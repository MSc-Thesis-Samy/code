use std::collections::HashMap;

pub type ActivationFunction = fn(f64) -> f64;

pub const SIGMOID: ActivationFunction = |x| 1. / (1. + (-4.9 * x).exp());
pub const IDENTITY: ActivationFunction = |x| x;

#[derive(Debug)]
pub struct NeuronInput {
    input_id: u32,
    weight: f64,
}

#[derive(Debug)]
pub struct Neuron {
    id: u32,
    inputs: Vec<NeuronInput>,
    activation: ActivationFunction,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    bias_id: Option<u32>,
    neurons: Vec<Neuron>, // Neurons are ordered by layer
}

impl NeuronInput {
    pub fn new(input_id: u32, weight: f64) -> NeuronInput {
        NeuronInput {
            input_id,
            weight,
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
            Neuron::new(3, vec![NeuronInput::new(1, 0.5), NeuronInput::new(2, 0.5)], IDENTITY),
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
            Neuron::new(3, vec![NeuronInput::new(1, 0.5), NeuronInput::new(2, 0.5)], IDENTITY),
            Neuron::new(4, vec![NeuronInput::new(3, 0.5)], IDENTITY),
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
            Neuron::new(3, vec![NeuronInput::new(1, 0.5), NeuronInput::new(2, 0.5), NeuronInput::new(4, 1.)], IDENTITY),
            Neuron::new(4, vec![], IDENTITY),
        ];
        let network = NeuralNetwork::new(input_ids, output_ids, Some(4), neurons);

        let inputs = vec![2., 2.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![3.]);
    }
}
