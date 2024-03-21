use crate::network::Network;
use crate::discrete_network::DiscreteNetwork;
use crate::vneuron::VNeuron;
use crate::discrete_vneuron::DiscreteVNeuron;
use crate::benchmarks::SphereClassificationProblem;

pub trait NeuroevolutionAlgorithm {
    fn optimization_step(&mut self, problem: &SphereClassificationProblem);
    fn optimize(&mut self, problem: &SphereClassificationProblem, n_iters: u32) {
        for _ in 0..n_iters {
            self.optimization_step(problem);
        }
    }
    fn optimize_cmaes(&mut self, problem: &SphereClassificationProblem);
    fn evaluate(&self, input: &Vec<f64>) -> bool;
}

pub enum Algorithm {
    DiscreteOneplusoneNA(DiscreteNetwork),
    ContinuousOneplusoneNA(Network),
    DiscreteBNA(DiscreteVNeuron),
    ContinuousBNA(VNeuron),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::ContinuousOneplusoneNA(network) => write!(f, "{}", network),
            Algorithm::DiscreteBNA(vneuron) => write!(f, "{}", vneuron),
            Algorithm::ContinuousBNA(vneuron) => write!(f, "{}", vneuron),
        }
    }
}

impl NeuroevolutionAlgorithm for Algorithm {
    fn optimize(&mut self, problem: &SphereClassificationProblem, n_iters: u32) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize(problem, n_iters),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize(problem, n_iters),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize(problem, n_iters),
        }
    }

    fn optimize_cmaes(&mut self, problem: &SphereClassificationProblem) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize_cmaes(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize_cmaes(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize_cmaes(problem),
        }
    }

    fn evaluate(&self, input: &Vec<f64>) -> bool {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.evaluate(input),
            Algorithm::ContinuousOneplusoneNA(network) => network.evaluate(input),
            Algorithm::DiscreteBNA(vneuron) => vneuron.evaluate(input),
            Algorithm::ContinuousBNA(vneuron) => vneuron.evaluate(input),
        }
    }

    fn optimization_step(&mut self, problem: &SphereClassificationProblem) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimization_step(problem),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimization_step(problem),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimization_step(problem),
        }
    }
}
