use neuroevolution::neat::*;
use neuroevolution::benchmarks::ClassificationProblem;
use neuroevolution::neuroevolution_algorithm::*;

fn main() {
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

    let mut neat = Neat::new(config);
    neat.optimize(&ClassificationProblem::Xor, 1500);
    println!("Fitness: {:.2}", neat.get_best_individual_fitness());
}
