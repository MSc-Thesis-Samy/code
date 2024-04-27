use neuroevolution::neural_network::*;
use neuroevolution::benchmarks::*;
use neuroevolution::neuroevolution_algorithm::Algorithm;
use neuroevolution::neuroevolution_algorithm::NeuroevolutionAlgorithm;

fn main() {
    let network = NeuralNetwork::new(
        vec![1, 2],
        vec![5],
        Some(3),
        vec![
            Neuron::new(4, vec![NeuronInput::new(1, None), NeuronInput::new(2, None), NeuronInput::new(3, None)], SIGMOID),
            Neuron::new(5, vec![NeuronInput::new(4, None), NeuronInput::new(3, None), NeuronInput::new(2, None), NeuronInput::new(3, None)], SIGMOID),
        ]
    );

    let problem = Benchmark::new(Problem::Xor);
    let mut alg = Algorithm::NeuralNetwork(network);
    alg.optimize_cmaes(&problem);
    println!("Fitness: {:.2}", problem.evaluate(&alg));
}
