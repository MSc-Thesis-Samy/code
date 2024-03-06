use neuroevolution::vneuron::*;
use neuroevolution::discrete_vneuron::*;
use neuroevolution::benchmarks::*;
use neuroevolution::traits::NeuroevolutionAlgorithm;

const N_ITERS: u32 = 5000;
const R: usize = 100;

fn main() {
    let mut vneuron = VNeuron::new(2);
    vneuron.optimize(half, N_ITERS);
    println!("half fitness: {:.2}", half(&vneuron));
    println!("Half: {}", vneuron);

    vneuron.optimize(quarter, N_ITERS);
    println!("quarter fitness: {:.2}", quarter(&vneuron));
    println!("Quarter: {}", vneuron);

    vneuron.optimize(two_quarters, N_ITERS);
    println!("two_quarters fitness: {:.2}", two_quarters(&vneuron));
    println!("Two Quarters: {}", vneuron);

    let mut dvneuron = DiscreteVNeuron::new(R, 2);
    dvneuron.optimize(half, N_ITERS);
    println!("half fitness: {:.2}", half(&dvneuron));
    println!("Half: {}", dvneuron);

    dvneuron.optimize(quarter, N_ITERS);
    println!("quarter fitness: {:.2}", quarter(&dvneuron));
    println!("Quarter: {}", dvneuron);

    dvneuron.optimize(two_quarters, N_ITERS);
    println!("two_quarters fitness: {:.2}", two_quarters(&dvneuron));
    println!("Two Quarters: {}", dvneuron);
}
