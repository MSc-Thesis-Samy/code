use neuroevolution::vneuron::*;
use neuroevolution::discrete_vneuron::*;
use neuroevolution::benchmarks::*;
use neuroevolution::traits::NeuroevolutionAlgorithm;

const N_ITERS: u32 = 5000;

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

    // vneuron.optimize(square, N_ITERS);
    // println!("square fitness: {:.2}", square(&vneuron));
    // println!("Square: {}", vneuron);

    let mut dvneuron = DiscreteVNeuron::new(2, 100);
    dvneuron.optimize(half, N_ITERS);
    println!("half fitness: {:.2}", half(&dvneuron));
    println!("Half: {}", dvneuron);

    // let vneuron = VNeuron::from_parameters(
    //     -0.5,
    //     vec![3. * PI / 2.],
    //     PI / 3.
    // );
}
