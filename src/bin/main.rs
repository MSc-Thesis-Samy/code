use clap::Parser;
use neuroevolution::cli::*;
use neuroevolution::vneuron::VNeuron;
use neuroevolution::discrete_vneuron::DiscreteVNeuron;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::{NeuroevolutionAlgorithm, Algorithm};
use neuroevolution::benchmarks::*;

fn main() {
    let cli = Cli::parse();

    match cli.problem {
        Problem::Half | Problem::Quarter | Problem::Twoquartes => {
            let dim = 2;
            let problem = match cli.problem {
                Problem::Half => half,
                Problem::Quarter => quarter,
                Problem::Twoquartes => two_quarters,
            };

            match cli.algorithm {
                AlgorithmType::Oneplusonena => {
                    match cli.continuous {
                        true => {
                            let mut network = Network::new(cli.neurons, dim);
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::ContinuousOneplusoneNA(&mut network)));
                            print!("Network: {}", network);
                        }
                        false => {
                            let mut network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::DiscreteOneplusoneNA(&mut network)));
                            print!("DiscreteNetwork: {}", network);
                        }
                    }
                },
                AlgorithmType::Bna => {
                    match cli.continuous {
                        true => {
                            let mut vneuron = VNeuron::new(dim);
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::ContinuousBNA(&mut vneuron)));
                            println!("VNeuron: {}", vneuron);
                        },
                        false => {
                            let mut vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::DiscreteBNA(&mut vneuron)));
                            println!("DiscreteVNeuron: {}", vneuron);
                        }
                    }
                }
            }
        }
    }
}
