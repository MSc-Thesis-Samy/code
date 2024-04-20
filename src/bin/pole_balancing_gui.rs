use std::fs::File;
use ggez::*;
use neuroevolution::neat::{Neat, Config};
use neuroevolution::neuroevolution_algorithm::{Algorithm, NeuroevolutionAlgorithm};
use neuroevolution::benchmarks::Benchmark;
use neuroevolution::pole_balancing_gui::State;

fn main() {
    let pole_balancing_state = neuroevolution::pole_balancing::State::default();

    let config = Config {
        population_size: 1000,
        n_inputs: 4,
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
        matching_weight: 3.0,
        champion_copy_threshold: 4,
        stagnation_threshold: 1500,
    };

    let neat = Neat::new(config);
    let mut alg = Algorithm::Neat(neat);
    let problem = Benchmark::PoleBalancing;
    println!("Optimizing algorithm...");
    alg.optimize(&problem, 20);
    println!("Fitness: {}", problem.evaluate(&alg));

    let state = State::new(pole_balancing_state, alg);

    let mut conf_file = File::open("gui_conf.toml").unwrap();
    let conf = conf::Conf::from_toml_file(&mut conf_file).unwrap();

    let cb = ContextBuilder::new("Neuroevolution", "Samy Haffoudhi") .default_conf(conf);
    let (ctx, event_loop) = cb.build().unwrap();

    event::run(ctx, event_loop, state);
}
