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

fn get_algorithm(algorithm_type: AlgorithmType, resolution: usize, neurons: usize, dim: usize, toml_config: &Option<String>) -> Algorithm {
    match algorithm_type {
        AlgorithmType::Neat => {
            let toml_config: &str = toml_config.as_deref().unwrap();
            let config: Config = toml::from_str(toml_config).unwrap();

            let neat = Neat::new(config);
            Algorithm::Neat(neat)
        }
        AlgorithmType::NeuralNetwork => {
            let toml_config: &str = toml_config.as_deref().unwrap();
            let network_config: NeuralNetworkConfig = toml::from_str(&toml_config).unwrap();

            let network = network_config.to_neural_network();
            Algorithm::NeuralNetwork(network)
        }
        AlgorithmType::Oneplusonena => {
            let network = DiscreteNetwork::new(resolution, neurons, dim);
            Algorithm::DiscreteOneplusoneNA(network)
        }
        AlgorithmType::Bna => {
            let vneuron = DiscreteVNeuron::new(resolution, dim);
            Algorithm::DiscreteBNA(vneuron)
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let dim = 2;
    let problem = Benchmark::new(cli.problem);
    let resolution = cli.resolution;
    let neurons = cli.neurons;
    let iterations = cli.iterations;
    let stagnation = cli.stagnation;
    let max_fitness_tol = cli.error_tol;

    let mut alg: Algorithm;

    let toml_config = if let Some(file) = cli.file {
        let mut config_file = File::open(file).unwrap();
        let mut toml_config = String::new();
        config_file.read_to_string(&mut toml_config).unwrap();
        Some(toml_config)
    } else {
        None
    };

    alg = get_algorithm(cli.algorithm, resolution, neurons, dim, &toml_config);

    if let Some(n_runs) = cli.test_runs {
        let results = (0..n_runs).into_par_iter().map(|_| {
            let mut alg = get_algorithm(cli.algorithm, resolution, neurons, dim, &toml_config);
            let problem = Benchmark::new(cli.problem);

            let start = Instant::now();
            let n_iters = alg.optimize_with_early_stopping(&problem, iterations, max_fitness_tol, stagnation);
            let elapsed = start.elapsed().as_secs_f64();

            (problem.evaluate(&alg), n_iters, elapsed)
        }).collect::<Vec<_>>();

        if let Some(output_path) = cli.output {
            let mut output_file = File::create(output_path).unwrap();
            writeln!(output_file, "fitness,iterations,cpu").unwrap();
            for (fitness, n_iters, elapsed) in results {
                writeln!(output_file, "{:.2},{},{:.3}", fitness, n_iters, elapsed).unwrap();
            }
        } else {
            let fitness_mean = results.iter().map(|(fitness, _, _)| fitness).sum::<f64>() / n_runs as f64;
            let n_iters_mean = results.iter().map(|(_, n_iters, _)| n_iters).sum::<u32>() / n_runs as u32;
            let elapsed_mean = results.iter().map(|(_, _, elapsed)| elapsed).sum::<f64>() / n_runs as f64;

            println!("Mean fitness: {:.2}", fitness_mean);
            println!("Mean iterations: {}", n_iters_mean);
            println!("Mean elapsed time: {:.3} s", elapsed_mean);
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
                    let _ = alg.optimize_with_early_stopping(&problem, iterations, max_fitness_tol, stagnation);
                    println!("Fitness: {}", problem.evaluate(&alg));

                    let pole_balancing_state = neuroevolution::pole_balancing::State::default();
                    let state = neuroevolution::pole_balancing_gui::State::new(pole_balancing_state, alg);
                    event::run(ctx, event_loop, state);
                }

                _ => {
                    let state = neuroevolution::gui::State::new(alg, problem, iterations);
                    event::run(ctx, event_loop, state);
                }
            }
        }

        false => {
            let n_iters = alg.optimize_with_early_stopping(&problem, cli.iterations, cli.error_tol, None);
            println!("Iterations: {}\nFitness: {:.2}", n_iters, problem.evaluate(&alg));
        }
    }
}
