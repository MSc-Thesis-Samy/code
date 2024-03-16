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
                            let network = Network::new(cli.neurons, dim);
                            let mut alg = Algorithm::ContinuousOneplusoneNA(network);
                            alg.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&alg));
                            print!("{}", alg);
                        }
                        false => {
                            let network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                            let mut alg = Algorithm::DiscreteOneplusoneNA(network);
                            alg.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&alg));
                            print!("{}", alg);
                        }
                    }
                },
                AlgorithmType::Bna => {
                    match cli.continuous {
                        true => {
                            let vneuron = VNeuron::new(dim);
                            let mut alg = Algorithm::ContinuousBNA(vneuron);
                            alg.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&alg));
                            print!("{}", alg);
                        },
                        false => {
                            let vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                            let mut alg = Algorithm::DiscreteBNA(vneuron);
                            alg.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&alg));
                            print!("{}", alg);
                        }
                    }
                }
            }
        }
    }
}
