use neuroevolution::vneuron::*;
use neuroevolution::bna::*;

fn main() {
    let mut vneuron = VNeuron::new(2);

    vneuron.optimize(half, 1000);
    println!("half fitness: {}", half(&vneuron));
    println!("Half: {}", vneuron);
}
