use std::fs::File;
use ggez::*;
use clap::Parser;
use neuroevolution::cli::*;
use neuroevolution::vneuron::VNeuron;
use neuroevolution::discrete_vneuron::DiscreteVNeuron;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::{NeuroevolutionAlgorithm, Algorithm};
use neuroevolution::benchmarks::*;
use neuroevolution::constants::*;
use neuroevolution::gui::*;

fn main() {
    let cli = Cli::parse();
    let mut alg: Algorithm;
    let dim = 2;

    let problem = match cli.problem {
        Problem::Half => ClassificationProblem::SphereProblem(SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS)),
        Problem::Quarter => ClassificationProblem::SphereProblem(SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS)),
        Problem::TwoQuarters => ClassificationProblem::SphereProblem(SphereClassificationProblem::TwoQuarters(UNIT_CIRCLE_STEPS)),
    };

    match cli.algorithm {
        AlgorithmType::Oneplusonena => {
            match cli.continuous {
                true => {
                    let network = Network::new(cli.neurons, dim);
                    alg = Algorithm::ContinuousOneplusoneNA(network);
                }
                false => {
                    let network = DiscreteNetwork::new(cli.resolution, cli.neurons, dim);
                    alg = Algorithm::DiscreteOneplusoneNA(network);
                }
            }
        },
        AlgorithmType::Bna => {
            match cli.continuous {
                true => {
                    let vneuron = VNeuron::new(dim);
                    alg = Algorithm::ContinuousBNA(vneuron);
                },
                false => {
                    let vneuron = DiscreteVNeuron::new(cli.resolution, dim);
                    alg = Algorithm::DiscreteBNA(vneuron);
                }
            }
        }
    }

    match cli.gui {
        true => {
            let state = State::new(alg, problem, N_ITERATIONS);
            let mut conf_file = File::open("gui_conf.toml").unwrap();
            let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();
            let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
            let (ctx, event_loop) = cb.build().unwrap();
            event::run(ctx, event_loop, state);
        }
        false => {
            alg.optimize(&problem, cli.iterations);

            println!("fitness: {:.2}", problem.evaluate(&alg));
            print!("{}", alg);
        }
    }
}
