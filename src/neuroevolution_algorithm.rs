use crate::network::Network;
use crate::discrete_network::DiscreteNetwork;
use crate::vneuron::VNeuron;
use crate::discrete_vneuron::DiscreteVNeuron;
use crate::benchmarks::ClassificationProblem;
use crate::neat::Neat;

pub trait NeuroevolutionAlgorithm {
    fn optimization_step(&mut self, problem: &ClassificationProblem);
    fn optimize(&mut self, problem: &ClassificationProblem, n_iters: u32) {
        for _ in 0..n_iters {
            self.optimization_step(problem);
        }
    }
    fn optimize_cmaes(&mut self, problem: &ClassificationProblem);
    fn evaluate(&self, input: &Vec<f64>) -> f64;
}

pub enum Algorithm {
    DiscreteOneplusoneNA(DiscreteNetwork),
    ContinuousOneplusoneNA(Network),
    DiscreteBNA(DiscreteVNeuron),
    ContinuousBNA(VNeuron),
    Neat(Neat),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::ContinuousOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::DiscreteBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::ContinuousBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::Neat(neat) => write!(f, "{:?}", neat), // TODO: Implement Display for Neat
        }
    }
}

impl NeuroevolutionAlgorithm for Algorithm {
    fn optimize(&mut self, problem: &ClassificationProblem, n_iters: u32) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize(problem, n_iters),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize(problem, n_iters),
            Algorithm::Neat(neat) => neat.optimize(problem, n_iters),
        }
    }

    fn optimize_cmaes(&mut self, problem: &ClassificationProblem) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize_cmaes(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize_cmaes(problem),
            Algorithm::Neat(neat) => neat.optimize_cmaes(problem),
        }
    }

    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.evaluate(input),
            Algorithm::ContinuousOneplusoneNA(network) => network.evaluate(input),
            Algorithm::DiscreteBNA(vneuron) => vneuron.evaluate(input),
            Algorithm::ContinuousBNA(vneuron) => vneuron.evaluate(input),
            Algorithm::Neat(neat) => neat.evaluate(input),
        }
    }

    fn optimization_step(&mut self, problem: &ClassificationProblem) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::Neat(neat) => neat.optimization_step(problem),
        }
    }
}
