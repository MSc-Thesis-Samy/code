use neuroevolution::neat::*;

fn xor(individual: &Individual) -> f32 {
    let inputs = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
    let outputs = vec![0., 1., 1., 0.];

    let mut distance_sum = 0.;
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        let result = individual.evaluate(input)[0];
        distance_sum += (result - output).abs();
    }

    4. - distance_sum
}

fn main() {
    let config = Config {
        population_size: 150,
        n_inputs: 2,
        n_outputs: 1,
        n_generations: 3000,
        evaluation_function: xor,
        weights_mean: 0.,
        weights_stddev: 6.,
        perturbation_stddev: 3.,
        survival_threshold: 0.25,
        connection_mutation_rate: 0.3,
        node_mutation_rate: 0.1,
        weight_mutation_rate: 0.8,
        similarity_threshold: 3.,
        excess_weight: 1.,
        disjoint_weight: 1.,
        matching_weight: 0.4,
    };

    let mut neat = Neat::new(config);
    neat.run();
}
