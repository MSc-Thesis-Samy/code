use crate::network::Network;
use crate::discrete_network::DiscreteNetwork;
use crate::vneuron::VNeuron;
use crate::discrete_vneuron::DiscreteVNeuron;

pub trait NeuroevolutionAlgorithm {
    fn optimization_step(&mut self, evaluation_function: fn(&Algorithm) -> f64);
    fn optimize(&mut self, evaluation_function: fn(&Algorithm) -> f64, n_iters: u32) {
        for _ in 0..n_iters {
            self.optimization_step(evaluation_function);
        }
    }

    // fn optimize_with_callback(&mut self, evaluation_function: fn(&Algorithm) -> f64, n_iters: u32)
    // where
    //     Self: std::fmt::Display
    // {
    //     for i in 0..n_iters {
    //         self.optimization_step(evaluation_function);
    //         println!("iteration: {}\nnetwork: {}", i, self);
    //     }
    // }

    fn optimize_cmaes(&mut self, evaluation_function: fn(&Algorithm) -> f64);
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
    fn optimize(&mut self, evaluation_function: fn(&Algorithm) -> f64, n_iters: u32) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize(evaluation_function, n_iters),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize(evaluation_function, n_iters),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize(evaluation_function, n_iters),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize(evaluation_function, n_iters),
        }
    }

    fn optimize_cmaes(&mut self, evaluation_function: fn(&Algorithm) -> f64) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimize_cmaes(evaluation_function),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimize_cmaes(evaluation_function),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimize_cmaes(evaluation_function),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimize_cmaes(evaluation_function),
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

    fn optimization_step(&mut self, evaluation_function: fn(&Algorithm) -> f64) {
        match self {
            Algorithm::DiscreteOneplusoneNA(network) => network.optimization_step(evaluation_function),
            Algorithm::ContinuousOneplusoneNA(network) => network.optimization_step(evaluation_function),
            Algorithm::DiscreteBNA(vneuron) => vneuron.optimization_step(evaluation_function),
            Algorithm::ContinuousBNA(vneuron) => vneuron.optimization_step(evaluation_function),
        }
    }
}
