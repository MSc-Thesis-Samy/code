use neuroevolution::vneuron::*;
use neuroevolution::bna::*;

fn main() {
    let mut vneuron = VNeuron::new(2);
    vneuron.optimize(half, 1000);
    println!("half fitness: {:.2}", half(&vneuron));
    println!("Half: {}", vneuron);

    // let angle = PI / 2.;
    // let bias = 0.;
    // let bend = PI / 2.;
    // let network = VNeuron::from_params(2, bias, vec![angle], bend);
    // println!("half fitness: {:.2}", half(&network));
    // println!("Half: {}", network);

    vneuron.optimize(quarter, 1000);
    println!("quarter fitness: {:.2}", quarter(&vneuron));
    println!("Quarter: {}", vneuron);

    vneuron.optimize(two_quarters, 1000);
    println!("two_quarters fitness: {:.2}", two_quarters(&vneuron));
    println!("Two Quarters: {}", vneuron);

    vneuron.optimize(square, 1000);
    println!("square fitness: {:.2}", square(&vneuron));
    println!("Square: {}", vneuron);
}
