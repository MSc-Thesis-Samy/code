use rand::prelude::*;

#[derive(Clone, Debug)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Clone, Debug)]
pub struct NodeGene {
    id: u32,
    node_type: NodeType,
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
    pub fn new(id: u32, node_type: NodeType) -> NodeGene {
        NodeGene { id, node_type }
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
}

impl Individual {
    pub fn new(genome: Genome) -> Individual {
        Individual {
            genome,
            fitness: 0.0,
        }
    }

    fn mutate(&mut self) {
        unimplemented!();
    }

    pub fn mutate_add_connection(&mut self) {
        unimplemented!();
    }

    pub fn mutate_add_node(&mut self, history: &mut History) {
        let new_node = NodeGene::new(history.nodes_nb + 1, NodeType::Hidden);
        history.nodes_nb += 1;
        self.genome.add_node(new_node.clone());

        // TODO always pick an enable connection?
        let connection = self.genome.connections.choose_mut(&mut thread_rng()).unwrap();
        connection.enabled = false;

        let in_to_new_node_connection = ConnectionGene::new(connection.in_node, new_node.id, 1.0, true, history.innovation + 1);
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

        Individual {
            genome: Genome {
                nodes,
                connections,
            },
            fitness: 0.0, // TODO: calculate fitness
        }
    }
}
