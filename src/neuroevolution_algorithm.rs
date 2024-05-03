use crate::network::Network;
use crate::discrete_network::DiscreteNetwork;
use crate::vneuron::VNeuron;
use crate::discrete_vneuron::DiscreteVNeuron;
use crate::benchmarks::Benchmark;
use crate::neat::{Neat, Individual};
use crate::neural_network::NeuralNetwork;

pub trait NeuroevolutionAlgorithm {
    fn optimization_step(&mut self, problem: &Benchmark);
    fn optimize(&mut self, problem: &Benchmark, n_iters: u32) {
        for _ in 0..n_iters {
            self.optimization_step(problem);
        }
    }
    fn optimize_cmaes(&mut self, problem: &Benchmark);
    fn evaluate(&self, input: &Vec<f64>) -> f64;
}

#[derive(Clone)]
pub enum Algorithm {
    DiscreteOneplusoneNA(DiscreteNetwork),
    ContinuousOneplusoneNA(Network),
    DiscreteBNA(DiscreteVNeuron),
    ContinuousBNA(VNeuron),
    Neat(Neat),
    NeatIndividual(Individual),
    NeuralNetwork(NeuralNetwork),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::ContinuousOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::DiscreteBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::ContinuousBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::Neat(neat) => write!(f, "{:?}", neat), // TODO: Implement Display for Neat
            Algorithm::NeuralNetwork(network) => write!(f, "{:?}", network),
            Algorithm::NeatIndividual(individual) => write!(f, "{:?}", individual),
        }
    }
}

impl NeuroevolutionAlgorithm for Algorithm {
    fn optimize(&mut self, problem: &Benchmark, n_iters: u32) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize(problem, n_iters),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize(problem, n_iters),
            Algorithm::Neat(neat) => neat.optimize(problem, n_iters),
            Algorithm::NeuralNetwork(network) => network.optimize(problem, n_iters),
            Algorithm::NeatIndividual(individual) => individual.optimize(problem, n_iters),
        }
    }

    fn optimize_cmaes(&mut self, problem: &Benchmark) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize_cmaes(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize_cmaes(problem),
            Algorithm::Neat(neat) => neat.optimize_cmaes(problem),
            Algorithm::NeuralNetwork(network) => network.optimize_cmaes(problem),
            Algorithm::NeatIndividual(individual) => individual.optimize_cmaes(problem),
        }
    }

    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.evaluate(input),
            Algorithm::ContinuousOneplusoneNA(network) => network.evaluate(input),
            Algorithm::DiscreteBNA(vneuron) => vneuron.evaluate(input),
            Algorithm::ContinuousBNA(vneuron) => vneuron.evaluate(input),
            Algorithm::Neat(neat) => neat.evaluate(input),
            Algorithm::NeuralNetwork(network) => network.evaluate(input),
            Algorithm::NeatIndividual(individual) => individual.evaluate(input),
        }
    }

    fn optimization_step(&mut self, problem: &Benchmark) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::Neat(neat) => neat.optimization_step(problem),
            Algorithm::NeuralNetwork(network) => network.optimization_step(problem),
            Algorithm::NeatIndividual(individual) => individual.optimization_step(problem),
        }
    }
}
