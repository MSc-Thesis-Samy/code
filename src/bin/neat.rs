use neuroevolution::neat::*;

fn main() {
    let node1 = NodeGene::new(1, NodeType::Input);
    let node2 = NodeGene::new(2, NodeType::Output);
    let connection = ConnectionGene::new(1, 2, 0.5, true, 1);
    let mut genome = Genome::new();
    genome.add_node(node1);
    genome.add_node(node2);
    genome.add_connection(connection);
    let individual1 = Individual::new(genome.clone());
    let node3 = NodeGene::new(3, NodeType::Output);
    let connection2 = ConnectionGene::new(1, 3, 0.3, true, 2);
    genome.add_node(node3);
    genome.add_connection(connection2);
    let individual2 = Individual::new(genome);
    let individual3 = Individual::crossover(&individual1, &individual2);
    println!("{:?}", individual1);
    println!("{:?}", individual2);
    println!("{:?}", individual3);
}
