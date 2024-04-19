use neuroevolution::benchmarks::*;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;

fn main() {
    let half = Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Half(UNIT_CIRCLE_STEPS)));
    let quarter = Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Quarter(UNIT_CIRCLE_STEPS)));
    let two_quarters = Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::TwoQuarters(UNIT_CIRCLE_STEPS)));
    let square = Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Square));
    let cube = Benchmark::Classification(ClassificationProblem::SphereProblem(SphereClassificationProblem::Cube));

    let network = Network::new(2, 2);
    let mut alg = Algorithm::ContinuousOneplusoneNA(network);
    alg.optimize(&half, N_ITERATIONS);
    println!("half fitness: {:.2}", half.evaluate(&alg));
    print!("{}", alg);
    alg.optimize(&quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter.evaluate(&alg));
    print!("{}", alg);
    alg.optimize(&two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters.evaluate(&alg));
    print!("{}", alg);
    alg.optimize(&square, N_ITERATIONS);
    println!("square fitness: {:.2}", square.evaluate(&alg));
    print!("{}", alg);

    let network = Network::new(4, 3);
    let mut alg = Algorithm::ContinuousOneplusoneNA(network);
    alg.optimize(&cube, N_ITERATIONS);
    println!("cube fitness: {:.2}", cube.evaluate(&alg));
    print!("{}", alg);

    let network = DiscreteNetwork::new(RESOLUTION, 1, 2);
    let mut alg = Algorithm::DiscreteOneplusoneNA(network);
    alg.optimize(&half, N_ITERATIONS);
    println!("half fitness: {:.2}", half.evaluate(&alg));
    print!("{}", alg);
    alg.optimize(&quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter.evaluate(&alg));
    print!("{}", alg);
    alg.optimize(&two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters.evaluate(&alg));
    print!("{}", alg);

    let network = Network::new(1, 2);
    let mut alg = Algorithm::ContinuousOneplusoneNA(network);
    alg.optimize_cmaes(&half);
    println!("half fitness: {:.2}", half.evaluate(&alg));
    print!("{}", alg);
}
