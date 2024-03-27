use rand::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Clone, Debug)]
pub struct NodeGene {
    id: u32,
    layer: NodeType,
}

#[derive(Clone, Debug)]
pub struct ConnectionGene {
    in_node: u32,
    out_node: u32,
    weight: f32,
    enabled: bool,
    innovation: u32,
}

#[derive(Clone, Debug)]
pub struct Genome {
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

#[derive(Clone, Debug)]
pub struct Individual {
    genome: Genome,
    fitness: f32,
}

pub struct History {
    innovation: u32,
    nodes_nb: u32,
}

impl NodeGene {
    pub fn new(id: u32, layer: NodeType) -> NodeGene {
        NodeGene { id, layer }
    }
}

impl ConnectionGene {
    pub fn new(in_node: u32, out_node: u32, weight: f32, enabled: bool, innovation: u32) -> ConnectionGene {
        ConnectionGene { in_node, out_node, weight, enabled, innovation }
    }
}

impl Genome {
    pub fn new() -> Genome {
        Genome {
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: NodeGene) {
        self.nodes.push(node);
    }

    pub fn add_connection(&mut self, connection: ConnectionGene) {
        self.connections.push(connection);
    }

    fn are_connected(&self, in_node: u32, out_node: u32) -> bool {
        // TODO improve time complexity
        self.connections.iter().any(|c| c.in_node == in_node && c.out_node == out_node)
    }
}

impl Individual {
    pub fn new(genome: Genome) -> Individual {
        Individual {
            genome,
            fitness: 0.0,
        }
    }

    pub fn mutate_add_connection(&mut self, history: &mut History) {
        let in_nodes = self.genome.nodes.iter().filter(|n| n.layer != NodeType::Output).collect::<Vec<_>>();
        let out_nodes = self.genome.nodes.iter().filter(|n| n.layer != NodeType::Input).collect::<Vec<_>>();

        // TODO only choose unconnected nodes?
        let in_node = in_nodes.choose(&mut thread_rng()).unwrap();
        let out_node = out_nodes.choose(&mut thread_rng()).unwrap();

        if (in_node.layer == out_node.layer) || self.genome.are_connected(in_node.id, out_node.id) {
            return;
        }

        let weight = 1.0; // TODO random weight
        let connection = ConnectionGene::new(in_node.id, out_node.id, weight, true, history.innovation + 1);
        history.innovation += 1;

        self.genome.add_connection(connection);
    }

    pub fn mutate_add_node(&mut self, history: &mut History) {
        let new_node = NodeGene::new(history.nodes_nb + 1, NodeType::Hidden);
        history.nodes_nb += 1;
        self.genome.add_node(new_node.clone());

        // TODO always pick an enable connection?
        let connection = self.genome.connections.choose_mut(&mut thread_rng()).unwrap();
        connection.enabled = false;

        let in_to_new_node_connection = ConnectionGene::new(connection.in_node, new_node.id, 1.0, true, history.innovation + 1);
        history.innovation += 1;
        let new_to_out_node_connection = ConnectionGene::new(new_node.id, connection.out_node, connection.weight, true, connection.innovation);
        self.genome.connections.push(in_to_new_node_connection); self.genome.connections.push(new_to_out_node_connection);
    }

    pub fn mutate_weights(&mut self) {
        unimplemented!();
    }

    pub fn crossover(parent1: &Individual, parent2: &Individual) -> Individual {
        let merge_nodes = |p1: &[NodeGene], p2: &[NodeGene]| -> Vec<NodeGene> {
            let mut merged_nodes = Vec::new();
            let mut iter1 = p1.iter();
            let mut iter2 = p2.iter();

            let mut node1 = iter1.next();
            let mut node2 = iter2.next();

            while let (Some(n1), Some(n2)) = (node1, node2) {
                if n1.id == n2.id {
                    merged_nodes.push(n1.clone());
                    node1 = iter1.next();
                    node2 = iter2.next();
                } else if n1.id < n2.id {
                    merged_nodes.push(n1.clone());
                    node1 = iter1.next();
                } else {
                    merged_nodes.push(n2.clone());
                    node2 = iter2.next();
                }
            }

            if let Some(n1) = node1 {
                merged_nodes.push(n1.clone());
                merged_nodes.extend_from_slice(&iter1.cloned().collect::<Vec<_>>());
            }
            else if let Some(n2) = node2 {
                merged_nodes.push(n2.clone());
                merged_nodes.extend_from_slice(&iter2.cloned().collect::<Vec<_>>());
            }

            merged_nodes
        };

        let merge_connections = |p1: &[ConnectionGene], p2: &[ConnectionGene]| -> Vec<ConnectionGene> {
            let mut merged_connections = Vec::new();
            let mut iter1 = p1.iter();
            let mut iter2 = p2.iter();

            let mut conn1 = iter1.next();
            let mut conn2 = iter2.next();

            while let (Some(c1), Some(c2)) = (conn1, conn2) {
                if c1.innovation == c2.innovation {
                    let gene = if rand::random() { c1.clone() } else { c2.clone() };
                    merged_connections.push(gene);
                    conn1 = iter1.next();
                    conn2 = iter2.next();
                } else if c1.innovation < c2.innovation {
                    if parent1.fitness > parent2.fitness {
                        merged_connections.push(c1.clone());
                    }
                    conn1 = iter1.next();
                } else {
                    if parent2.fitness > parent1.fitness {
                        merged_connections.push(c2.clone());
                    }
                    conn2 = iter2.next();
                }
            }

            if parent1.fitness > parent2.fitness {
                if let Some(c1) = conn1 {
                    merged_connections.push(c1.clone());
                    merged_connections.extend_from_slice(&iter1.cloned().collect::<Vec<_>>());
                }
            }
            else {
                if let Some(c2) = conn2 {
                    merged_connections.push(c2.clone());
                    merged_connections.extend_from_slice(&iter2.cloned().collect::<Vec<_>>());
                }
            }

            merged_connections
        };

        let nodes = merge_nodes(&parent1.genome.nodes, &parent2.genome.nodes);
        let connections = merge_connections(&parent1.genome.connections, &parent2.genome.connections);

        println!("{:?}", connections);

        Individual {
            genome: Genome {
                nodes,
                connections,
            },
            fitness: 0.0, // TODO: calculate fitness
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover() {
        let node1 = NodeGene::new(1, NodeType::Input);
        let node2 = NodeGene::new(2, NodeType::Input);
        let node3 = NodeGene::new(3, NodeType::Input);
        let node4 = NodeGene::new(4, NodeType::Output);
        let node5 = NodeGene::new(5, NodeType::Hidden);
        let node6 = NodeGene::new(6, NodeType::Hidden);

        let conn_1_4 = ConnectionGene::new(1, 4, 1., true, 1);
        let conn_2_4 = ConnectionGene::new(2, 4, 1., false, 2);
        let conn_3_4 = ConnectionGene::new(3, 4, 1., true, 3);
        let conn_2_5 = ConnectionGene::new(2, 5, 1., true, 4);
        let conn_5_4 = ConnectionGene::new(5, 4, 1., true, 5);
        let conn_1_5 = ConnectionGene::new(1, 5, 1., true, 8);
        let conn_5_4_bis = ConnectionGene::new(5, 4, 1., false, 5);
        let conn_5_6 = ConnectionGene::new(5, 6, 1., true, 6);
        let conn_6_4 = ConnectionGene::new(6, 4, 1., true, 7);
        let conn_3_5 = ConnectionGene::new(3, 5, 1., true, 9);
        let conn_1_6 = ConnectionGene::new(1, 6, 1., true, 10);

        let mut genome1 = Genome::new();
        genome1.add_node(node1.clone());
        genome1.add_node(node2.clone());
        genome1.add_node(node3.clone());
        genome1.add_node(node4.clone());
        genome1.add_node(node5.clone());
        genome1.add_connection(conn_1_4.clone());
        genome1.add_connection(conn_2_4.clone());
        genome1.add_connection(conn_3_4.clone());
        genome1.add_connection(conn_2_5.clone());
        genome1.add_connection(conn_5_4.clone());
        genome1.add_connection(conn_1_5.clone());

        let mut genome2 = Genome::new();
        genome2.add_node(node1);
        genome2.add_node(node2);
        genome2.add_node(node3);
        genome2.add_node(node4);
        genome2.add_node(node5);
        genome2.add_node(node6);
        genome2.add_connection(conn_1_4);
        genome2.add_connection(conn_2_4);
        genome2.add_connection(conn_3_4);
        genome2.add_connection(conn_2_5);
        genome2.add_connection(conn_5_4_bis);
        genome2.add_connection(conn_5_6);
        genome2.add_connection(conn_6_4);
        genome2.add_connection(conn_3_5);
        genome2.add_connection(conn_1_6);

        let mut parent1 = Individual::new(genome1);
        let mut parent2 = Individual::new(genome2);

        parent1.fitness = 1.0;
        parent2.fitness = 0.;
        let child = Individual::crossover(&parent1, &parent2);
        assert_eq!(child.genome.nodes.len(), 6);
        assert_eq!(child.genome.connections.len(), 6);

        parent1.fitness = 0.;
        parent2.fitness = 1.0;
        let child = Individual::crossover(&parent1, &parent2);
        assert_eq!(child.genome.nodes.len(), 6);
        assert_eq!(child.genome.connections.len(), 9);
    }

    #[test]
    fn test_mutate_add_node() {
        let node1 = NodeGene::new(1, NodeType::Input);
        let node2 = NodeGene::new(2, NodeType::Output);
        let connection = ConnectionGene::new(1, 2, 0.5, true, 1);
        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        genome.add_connection(connection);
        let mut individual = Individual::new(genome);

        let mut history = History { innovation: 1, nodes_nb: 2 };
        individual.mutate_add_node(&mut history);

        assert_eq!(individual.genome.nodes.len(), 3);
        assert_eq!(individual.genome.connections.len(), 3);
        assert_eq!(history.nodes_nb, 3);
        assert_eq!(history.innovation, 2);

        let new_node = individual.genome.nodes.iter().find(|n| n.id == 3).unwrap();
        assert_eq!(new_node.layer, NodeType::Hidden);

        let old_connection = individual.genome.connections.iter().find(|c| c.in_node == 1 && c.out_node == 2).unwrap();
        let in_new_node_connection = individual.genome.connections.iter().find(|c| c.in_node == 1 && c.out_node == 3).unwrap();
        let new_out_node_connection = individual.genome.connections.iter().find(|c| c.in_node == 3 && c.out_node == 2).unwrap();
        assert!(!old_connection.enabled);
        assert_eq!(in_new_node_connection.weight, 1.);
        assert_eq!(new_out_node_connection.weight, 0.5);
    }

    #[test]
    fn test_mutate_add_connection() {
        let node1 = NodeGene::new(1, NodeType::Input);
        let node2 = NodeGene::new(2, NodeType::Output);
        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        let mut individual = Individual::new(genome);

        let mut history = History { innovation: 0, nodes_nb: 2 };
        individual.mutate_add_connection(&mut history);

        assert_eq!(individual.genome.nodes.len(), 2);
        assert_eq!(individual.genome.connections.len(), 1);
        assert_eq!(history.innovation, 1);

        let new_connection = individual.genome.connections.iter().find(|c| c.innovation == 1).unwrap();
        assert_eq!(new_connection.in_node, 1);
        assert_eq!(new_connection.out_node, 2);
    }
}
