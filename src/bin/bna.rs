use neuroevolution::vneuron::*;
use neuroevolution::discrete_vneuron::*;
use neuroevolution::benchmarks::*;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;

fn main() {
    let vneuron = VNeuron::new(2);
    let mut alg = Algorithm::ContinuousBNA(vneuron);
    alg.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&alg));
    println!("Half: {}", alg);
    alg.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&alg));
    println!("Half: {}", alg);
    alg.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&alg));
    println!("Half: {}", alg);

    let dvneuron = DiscreteVNeuron::new(RESOLUTION, 2);
    let mut alg = Algorithm::DiscreteBNA(dvneuron);
    alg.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&alg));
    println!("Half: {}", alg);
    alg.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&alg));
    println!("Half: {}", alg);
    alg.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&alg));
    println!("Half: {}", alg);
}
