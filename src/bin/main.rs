use std::time::Instant;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::io::Write;
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
use neuroevolution::neural_network::NeuralNetworkConfig;

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
        AlgorithmType::NeuralNetwork => {
            let config_file_path = match cli.file {
                Some(file) => file,
                None => panic!("No configuration file provided"),
            };

            let mut network_config_file = File::open(config_file_path).unwrap();
            let mut toml_config = String::new();
            network_config_file.read_to_string(&mut toml_config).unwrap();

            let network_config: NeuralNetworkConfig = toml::from_str(&toml_config).unwrap();

            let network = network_config.to_neural_network();
            alg = Algorithm::NeuralNetwork(network);
        }
    }

    if let Some(n_runs) = cli.test_runs {
        let (fitness_mean, elapsed_mean) = (0..n_runs).into_par_iter().map(|_| {
            let mut algorithm = alg.clone(); // TODO initialize with different initial values
            let problem = Benchmark::new(cli.problem);

            let start = Instant::now();
            algorithm.optimize(&problem, cli.iterations);
            let elapsed = start.elapsed().as_secs_f64();

            (problem.evaluate(&algorithm), elapsed)
        }).reduce(|| (0., 0.), |a, b| (a.0 + b.0, a.1 + b.1));

        if let Some(output_path) = cli.output {
            let mut output_file = File::create(output_path).unwrap();
            writeln!(output_file, "Mean fitness: {:.2}", fitness_mean / n_runs as f64).unwrap();
            writeln!(output_file, "Mean elapsed time: {:.3} s", elapsed_mean / n_runs as f64).unwrap();
        } else {
            println!("Mean fitness: {:.2}", fitness_mean / n_runs as f64);
            println!("Mean elapsed time: {:.3} s", elapsed_mean / n_runs as f64);
        }

        return;
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
