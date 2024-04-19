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
use neuroevolution::neat::*;

fn main() {
    let cli = Cli::parse();
    let mut alg: Algorithm;
    let dim = 2;

    let problem = match cli.problem {
        Problem::Half => Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS))),
        Problem::Quarter => Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS))),
        Problem::TwoQuarters => Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::TwoQuarters(UNIT_CIRCLE_STEPS))),
        Problem::Xor => Benchmark::Classification(ClassificationProblem::Xor),
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
        AlgorithmType::Neat => {
            let config = Config {
                population_size: 150,
                n_inputs: 2,
                n_outputs: 1,
                weights_mean: 0.,
                weights_stddev: 0.8,
                perturbation_stddev: 0.2,
                new_weight_probability: 0.1,
                enable_probability: 0.25,
                survival_threshold: 0.25,
                connection_mutation_rate: 0.3,
                node_mutation_rate: 0.03,
                weight_mutation_rate: 0.8,
                similarity_threshold: 15.0,
                excess_weight: 1.,
                disjoint_weight: 1.,
                matching_weight: 0.3,
                champion_copy_threshold: 5,
                stagnation_threshold: 1500,
            };

            let neat = Neat::new(config);
            alg = Algorithm::Neat(neat);
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
        }
    }
}
