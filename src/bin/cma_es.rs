use cmaes::fmax;
use neuroevolution::cma_es::*;
use neuroevolution::network::*;

fn main() {
    const N: usize = 2;

    let solution = fmax(half::<N>, vec![0.; N * 2], 1.);
    let network: Network<N,2> = get_network(&solution.point);
    let fitness = solution.value;
    println!("half Fitness: {}", fitness);
    println!("{}", network);
}
