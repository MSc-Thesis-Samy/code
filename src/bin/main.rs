use clap::Parser;
use neuroevolution::cli::*;
use neuroevolution::vneuron::VNeuron;
use neuroevolution::discrete_vneuron::DiscreteVNeuron;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::traits::NeuroevolutionAlgorithm;
use neuroevolution::benchmarks::*;

fn main() {
    let cli = Cli::parse();

    match cli.problem {
        Problem::Half | Problem::Quarter | Problem::Twoquartes => {
            let dim = 2;

            match cli.algorithm {
                Algorithm::Oneplusonena => {
                    match cli.continuous {
                        true => {
                            let mut network = Network::new(cli.neurons, dim);
                            let problem = match cli.problem {
                                Problem::Half => half::<Network>,
                                Problem::Quarter => quarter::<Network>,
                                Problem::Twoquartes => two_quarters::<Network>,
                            };
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&network));
                            print!("Network: {}", network);
                        }
                        false => {
                            let mut network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                            let problem = match cli.problem {
                                Problem::Half => half::<DiscreteNetwork>,
                                Problem::Quarter => quarter::<DiscreteNetwork>,
                                Problem::Twoquartes => two_quarters::<DiscreteNetwork>,
                            };
                            network.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&network));
                            print!("DiscreteNetwork: {}", network);
                        }
                    }
                },
                Algorithm::Bna => {
                    match cli.continuous {
                        true => {
                            let mut vneuron = VNeuron::new(dim);
                            let problem = match cli.problem {
                                Problem::Half => half::<VNeuron>,
                                Problem::Quarter => quarter::<VNeuron>,
                                Problem::Twoquartes => two_quarters::<VNeuron>,
                            };
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&vneuron));
                            println!("VNeuron: {}", vneuron);
                        },
                        false => {
                            let mut vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                            let problem = match cli.problem {
                                Problem::Half => half::<DiscreteVNeuron>,
                                Problem::Quarter => quarter::<DiscreteVNeuron>,
                                Problem::Twoquartes => two_quarters::<DiscreteVNeuron>,
                            };
                            vneuron.optimize(problem, cli.iterations);
                            println!("fitness: {:.2}", problem(&vneuron));
                            println!("DiscreteVNeuron: {}", vneuron);
                        }
                    }
                }
            }
        }
    }
}
