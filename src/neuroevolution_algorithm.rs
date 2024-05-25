use crate::network::Network;
use crate::discrete_network::DiscreteNetwork;
use crate::discrete_vnetwork::DiscreteVNetwork;
use crate::vneuron::VNeuron;
use crate::benchmarks::Benchmark;
use crate::neat::{Neat, Individual};
use crate::neural_network::NeuralNetwork;
use crate::constants::MAX_FITNESS;

pub trait NeuroevolutionAlgorithm {
    fn optimization_step(&mut self, problem: &Benchmark);
    fn optimize_cmaes(&mut self, problem: &Benchmark);
    fn evaluate(&self, input: &Vec<f64>) -> f64;

    fn optimize(&mut self, problem: &Benchmark, n_iters: u32) {
        for _ in 0..n_iters {
            self.optimization_step(problem);
        }
    }

    fn optimize_with_early_stopping(&mut self, problem: &Benchmark, max_iters: u32, fitness_tol: f64, max_stagnation: Option<u32>) -> u32 where Self: Sized {
        let mut best_fitness = 0.0;
        let mut stagnation = 0;
        let mut iters = 0;

        loop {
            self.optimization_step(problem);
            iters += 1;

            let fitness = problem.evaluate(self);
            if fitness > best_fitness {
                best_fitness = fitness;
                stagnation = 0;
            } else {
                stagnation += 1;
            }

            if (MAX_FITNESS - best_fitness).abs() < fitness_tol {
                break;
            }

            if let Some(max_stagnation) = max_stagnation {
                if stagnation >= max_stagnation {
                    break;
                }
            }

            if iters >= max_iters {
                break;
            }
        }

        iters
    }
}

#[derive(Clone)]
pub enum Algorithm {
    DiscreteOneplusoneNA(DiscreteNetwork),
    ContinuousOneplusoneNA(Network),
    DiscreteBNA(DiscreteVNetwork),
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
            Algorithm::DiscreteBNA(vnetwork) => write!(f, "{}", vnetwork),
            Algorithm::ContinuousBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::Neat(neat) => write!(f, "{:?}", neat), // TODO: Implement Display for Neat
            Algorithm::NeuralNetwork(network) => write!(f, "{:?}", network), // TODO: Implement Display for NeuralNetwork
            Algorithm::NeatIndividual(individual) => write!(f, "{:?}", individual),
        }
    }
}

impl NeuroevolutionAlgorithm for Algorithm {
    fn optimize(&mut self, problem: &Benchmark, n_iters: u32) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::DiscreteBNA(vnetwork) => vnetwork.optimize(problem, n_iters),
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
            Algorithm::DiscreteBNA(vnetwork) => vnetwork.optimize_cmaes(problem),
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
            Algorithm::DiscreteBNA(vnetwork) => vnetwork.evaluate(input),
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
            Algorithm::DiscreteBNA(vnetwork) => vnetwork.optimization_step(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::Neat(neat) => neat.optimization_step(problem),
            Algorithm::NeuralNetwork(network) => network.optimization_step(problem),
            Algorithm::NeatIndividual(individual) => individual.optimization_step(problem),
        }
    }

    fn optimize_with_early_stopping(&mut self, problem: &Benchmark, max_iters: u32, fitness_tol: f64, max_stagnation: Option<u32>) -> u32 where Self: Sized {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::DiscreteBNA(vnetwork) => vnetwork.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::Neat(neat) => neat.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::NeuralNetwork(network) => network.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
            Algorithm::NeatIndividual(individual) => individual.optimize_with_early_stopping(problem, max_iters, fitness_tol, max_stagnation),
        }
    }
}
