use std::fs::File;
use std::io::Read;
use toml;
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
use neuroevolution::neat::*;

fn main() {
    let cli = Cli::parse();
    let mut alg: Algorithm;
    let dim = 2;

    let problem = Benchmark::new(cli.problem);

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
            let config_file_path = match cli.file {
                Some(file) => file,
                None => panic!("No configuration file provided"),
            };

            let mut neat_config_file = File::open(config_file_path).unwrap();
            let mut toml_config = String::new();
            neat_config_file.read_to_string(&mut toml_config).unwrap();
            let config: Config = toml::from_str(&toml_config).unwrap();

            let neat = Neat::new(config);
            alg = Algorithm::Neat(neat);
        }
    }

    match cli.gui {
        true => {
            let mut conf_file = File::open("gui_conf.toml").unwrap();
            let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();
            let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
            let (ctx, event_loop) = cb.build().unwrap();

            match problem {
                Benchmark::PoleBalancing => {
                    println!("Evolving algorithm...");
                    alg.optimize(&problem, cli.iterations);
                    println!("Fitness: {}", problem.evaluate(&alg));

                    let pole_balancing_state = neuroevolution::pole_balancing::State::default();
                    let state = neuroevolution::pole_balancing_gui::State::new(pole_balancing_state, alg);
                    event::run(ctx, event_loop, state);
                }

                _ => {
                    let state = neuroevolution::gui::State::new(alg, problem, N_ITERATIONS);
                    event::run(ctx, event_loop, state);
                }
            }
        }

        false => {
            alg.optimize(&problem, cli.iterations);

            println!("fitness: {:.2}", problem.evaluate(&alg));
        }
    }
}
