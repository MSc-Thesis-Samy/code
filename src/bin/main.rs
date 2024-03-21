use clap::Parser;
use neuroevolution::cli::*;
use neuroevolution::vneuron::VNeuron;
use neuroevolution::discrete_vneuron::DiscreteVNeuron;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::{NeuroevolutionAlgorithm, Algorithm};
use neuroevolution::benchmarks::*;
use neuroevolution::constants::*;

fn main() {
    let cli = Cli::parse();

    match cli.problem {
        Problem::Half | Problem::Quarter | Problem::Twoquartes => {
            let dim = 2;
            let problem = match cli.problem {
                Problem::Half => SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS),
                Problem::Quarter => SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS),
                Problem::Twoquartes => SphereClassificationProblem::TwoQuarters(UNIT_CIRCLE_STEPS),
            };

            match cli.algorithm {
                AlgorithmType::Oneplusonena => {
                    match cli.continuous {
                        true => {
                            let network = Network::new(cli.neurons, dim);
                            let mut alg = Algorithm::ContinuousOneplusoneNA(network);
                            alg.optimize(&problem, cli.iterations);
                            println!("fitness: {:.2}", problem.evaluate(&alg));
                            print!("{}", alg);
                        }
                        false => {
                            let network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                            let mut alg = Algorithm::DiscreteOneplusoneNA(network);
                            alg.optimize(&problem, cli.iterations);
                            println!("fitness: {:.2}", problem.evaluate(&alg));
                            print!("{}", alg);
                        }
                    }
                },
                AlgorithmType::Bna => {
                    match cli.continuous {
                        true => {
                            let vneuron = VNeuron::new(dim);
                            let mut alg = Algorithm::ContinuousBNA(vneuron);
                            alg.optimize(&problem, cli.iterations);
                            println!("fitness: {:.2}", problem.evaluate(&alg));
                            print!("{}", alg);
                        },
                        false => {
                            let vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                            let mut alg = Algorithm::DiscreteBNA(vneuron);
                            alg.optimize(&problem, cli.iterations);
                            println!("fitness: {:.2}", problem.evaluate(&alg));
                            print!("{}", alg);
                        }
                    }
                }
            }
        }
    }
}
