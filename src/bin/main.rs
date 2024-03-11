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
                Problem::Half => half::<Algorithm>,
                Problem::Quarter => quarter::<Algorithm>,
                Problem::Twoquartes => two_quarters::<Algorithm>,
            };

            match cli.algorithm {
                AlgorithmType::Oneplusonena => {
                    match cli.continuous {
                        true => {
                            let mut network = Network::new(cli.neurons, dim);
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::ContinuousOneplusoneNA(network.clone())));
                            print!("Network: {}", network);
                        }
                        false => {
                            let mut network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::DiscreteOneplusoneNA(network.clone())));
                            print!("DiscreteNetwork: {}", network);
                        }
                    }
                },
                AlgorithmType::Bna => {
                    match cli.continuous {
                        true => {
                            let mut vneuron = VNeuron::new(dim);
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::ContinuousBNA(vneuron.clone())));
                            println!("VNeuron: {}", vneuron);
                        },
                        false => {
                            let mut vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&Algorithm::DiscreteBNA(vneuron.clone())));
                            println!("DiscreteVNeuron: {}", vneuron);
                        }
                    }
                }
            }
        }
    }
}
