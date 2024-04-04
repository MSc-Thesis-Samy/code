#![allow(dead_code)] // TODO remove

use rand::prelude::*;
use rand_distr::Normal;
use crate::neural_network::*;

type EvaluationFunction = fn(&Individual) -> f32;
type Population = Vec<Individual>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}

#[derive(Clone, Debug)]
struct NodeGene {
    id: u32,
    layer: NodeType,
    activation: ActivationFunction,
}

#[derive(Clone, Debug)]
struct ConnectionGene {
    in_node: u32,
    out_node: u32,
    weight: f32,
    enabled: bool,
    innovation: u32,
}

#[derive(Clone, Debug)]
struct Genome {
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

#[derive(Clone, Debug)]
pub struct Individual {
    genome: Genome,
    fitness: f32,
}

#[derive(Debug)]
enum Mutation {
    NewConnection(ConnectionGene),
    NewNode(NodeGene, ConnectionGene, ConnectionGene),
    // TODO Add weight perturbation?
}

#[derive(Debug)]
struct History {
    innovation: u32,
    nodes_nb: u32,
    mutations: Vec<(Mutation, u32)>,
    generation: u32,
}

pub struct Config {
    pub population_size: u32,
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub n_generations: u32,
    pub evaluation_function: EvaluationFunction,
    pub weights_mean: f32,
    pub weights_stddev: f32,
    pub perturbation_stddev: f32,
    pub survival_threshold: f32,
    pub connection_mutation_rate: f32,
    pub node_mutation_rate: f32,
    pub weight_mutation_rate: f32,
}

pub struct Neat {
    population: Population,
    history: History,
    config: Config,
}

impl NodeGene {
    fn new(id: u32, layer: NodeType, activation: ActivationFunction) -> NodeGene {
        NodeGene { id, layer, activation }
    }
}

impl ConnectionGene {
    fn new(in_node: u32, out_node: u32, weight: f32, enabled: bool, innovation: u32) -> ConnectionGene {
        ConnectionGene { in_node, out_node, weight, enabled, innovation }
    }
}

impl Genome {
    fn new() -> Genome {
        Genome {
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    fn add_node(&mut self, node: NodeGene) {
        self.nodes.push(node);
    }

    fn add_connection(&mut self, connection: ConnectionGene) {
        self.connections.push(connection);
    }

    fn are_connected(&self, in_node: u32, out_node: u32) -> bool {
        // TODO improve time complexity
        self.connections.iter().any(|c| c.in_node == in_node && c.out_node == out_node)
    }
}

impl Individual {
    fn new(genome: Genome) -> Individual {
        Individual {
            genome,
            fitness: 0.0,
        }
    }

    fn mutate_add_connection(&mut self, history: &mut History, distribution: &Normal<f32>) {
        let in_nodes = self.genome.nodes.iter().filter(|n| n.layer != NodeType::Output).collect::<Vec<_>>();
        let out_nodes = self.genome.nodes.iter().filter(|n| n.layer != NodeType::Input && n.layer != NodeType::Bias).collect::<Vec<_>>();

        // TODO only choose unconnected nodes?
        let in_node = in_nodes.choose(&mut thread_rng()).unwrap();
        let out_node = out_nodes.choose(&mut thread_rng()).unwrap();
        let weight = distribution.sample(&mut thread_rng());

        if (in_node.layer == out_node.layer) || self.genome.are_connected(in_node.id, out_node.id) {
            return;
        }

        for (mutation, generation) in history.mutations.iter().rev() {
            if *generation < history.generation {
                break;
            }

            match mutation {
                Mutation::NewConnection(other_connection) => {
                    if other_connection.in_node == in_node.id && other_connection.out_node == out_node.id {
                        let connection = ConnectionGene::new(in_node.id, out_node.id, weight, true, other_connection.innovation);

                        self.genome.add_connection(connection.clone());
                        history.mutations.push((Mutation::NewConnection(connection), history.generation));
                        return;
                    }
                }
                _ => continue
            }
        }

        history.innovation += 1;
        let connection = ConnectionGene::new(in_node.id, out_node.id, weight, true, history.innovation);

        self.genome.add_connection(connection.clone());
        history.mutations.push((Mutation::NewConnection(connection), history.generation));
    }

    fn mutate_add_node(&mut self, history: &mut History) {
        // TODO always pick an enabled connection?
        let connection = match self.genome.connections.choose_mut(&mut thread_rng()) {
            Some(connection) => connection,
            None => return
        };

        connection.enabled = false;

        for (mutation, generation) in history.mutations.iter().rev() {
            if *generation < history.generation {
                break;
            }

            match mutation {
                Mutation::NewNode(new_node, in_new_connection , new_out_connection) => {
                    if in_new_connection.in_node == connection.in_node && new_out_connection.out_node == connection.out_node {
                        self.genome.add_node(new_node.clone());
                        self.genome.connections.push(in_new_connection.clone());
                        self.genome.connections.push(new_out_connection.clone());
                        history.mutations.push((Mutation::NewNode(new_node.clone(), in_new_connection.clone(), new_out_connection.clone()), history.generation));
                        return;
                    }
                }
                _ => continue
            }
        }

        let new_node = NodeGene::new(history.nodes_nb + 1, NodeType::Hidden, SIGMOID ); // TODO get from config
        let in_new_connection = ConnectionGene::new(connection.in_node, new_node.id, 1., true, history.innovation + 1);
        let new_out_connection = ConnectionGene::new(new_node.id, connection.out_node, connection.weight, true, history.innovation + 2);
        history.nodes_nb += 1;
        history.innovation += 2;
        self.genome.add_node(new_node.clone());
        self.genome.connections.push(in_new_connection.clone());
        self.genome.connections.push(new_out_connection.clone());
        history.mutations.push((Mutation::NewNode(new_node.clone(), in_new_connection.clone(), new_out_connection.clone()), history.generation));
    }

    fn mutate_weights(&mut self, weights_distribution: &Normal<f32>, perturbation_distribution: &Normal<f32>) {
        // TODO set to random value
        for connection in self.genome.connections.iter_mut() {
            connection.weight += perturbation_distribution.sample(&mut thread_rng());
        }
    }

    fn crossover(parent1: &Individual, parent2: &Individual) -> Individual {
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

        Individual::new(Genome { nodes, connections })
    }

    fn to_neural_network(&self) -> NeuralNetwork {
        let (input_ids, input_activations): (Vec<u32>, Vec<ActivationFunction>) = self.genome.nodes.iter().filter(|n| n.layer == NodeType::Input).map(|n| (n.id, n.activation)).unzip();
        let (output_ids, output_activations): (Vec<u32>, Vec<ActivationFunction>) = self.genome.nodes.iter().filter(|n| n.layer == NodeType::Output).map(|n| (n.id, n.activation)).unzip();
        let mut input_activations = input_activations.iter();
        let mut output_activations = output_activations.iter();
        let bias  = self.genome.nodes.iter().find_map(|n| if n.layer == NodeType::Bias { Some((n.id, n.activation)) } else { None });

        let mut neurons = Vec::new();

        // Add input neurons
        for input_id in input_ids.iter() {
            let neuron = Neuron::new(*input_id, Vec::new(), *input_activations.next().unwrap());
            neurons.push(neuron);
        }

        // Add hidden neurons
        for node in self.genome.nodes.iter().filter(|n| n.layer == NodeType::Hidden) {
            let inputs = self.genome.connections.iter().filter(|c| c.out_node == node.id && c.enabled).map(|c| NeuronInput::new(c.in_node, c.weight)).collect::<Vec<_>>();
            let neuron = Neuron::new(node.id, inputs, node.activation);
            neurons.push(neuron);
        }

        // Add output neurons
        for output_id in output_ids.iter() {
            let inputs = self.genome.connections.iter().filter(|c| c.out_node == *output_id && c.enabled).map(|c| NeuronInput::new(c.in_node, c.weight)).collect::<Vec<_>>();
            let neuron = Neuron::new(*output_id, inputs, *output_activations.next().unwrap());
            neurons.push(neuron);
        }

        // Add bias neuron if it exists
        if let Some((bias_id, bias_activation)) = bias {
            let inputs = self.genome.connections.iter().filter(|c| c.out_node == bias_id && c.enabled).map(|c| NeuronInput::new(c.in_node, c.weight)).collect::<Vec<_>>();
            let neuron = Neuron::new(bias_id, inputs, bias_activation);
            neurons.push(neuron);
        }

        NeuralNetwork::new(input_ids, output_ids, bias.map(|(id, _)| id), neurons)
    }

    pub fn evaluate(&self, input: &Vec<f32>) -> Vec<f32> {
        let network = self.to_neural_network();
        network.feed_forward(input)
    }
}

impl History {
    fn new(n_neurons: u32, n_connections: u32, mutations: Vec<(Mutation, u32)>) -> History {
        History {
            innovation: n_connections,
            nodes_nb: n_neurons,
            mutations,
            generation: 0,
        }
    }
}

impl Neat {
    pub fn new(config: Config) -> Neat {
        let weights_distribution = Normal::new(config.weights_mean, config.weights_stddev).unwrap();
        let (history, population) = Self::get_initial_state(config.n_inputs, config.n_outputs, config.population_size as usize, &weights_distribution);

        Neat {
            population,
            history,
            config,
        }
    }

    fn get_initial_state(n_inputs: u32, n_outputs: u32, population_size: usize, weights_distribution: &Normal<f32>) -> (History, Population) {
        let get_initial_individual = |n_inputs, n_outputs, distributions: &Normal<f32>| {
            let mut genome = Genome::new();

            for i in 1..=n_inputs {
                let node = NodeGene::new(i, NodeType::Input, IDENTITY);
                genome.add_node(node);
            }

            for i in 1..=n_outputs {
                let node = NodeGene::new(i + n_inputs, NodeType::Output, SIGMOID); // TODO get from config?
                genome.add_node(node);
            }

            let bias = NodeGene::new(n_inputs + n_outputs + 1, NodeType::Bias, IDENTITY);
            genome.add_node(bias);

            // input -> output connections
            for i in 1..=n_inputs {
                for j in 1..=n_outputs {
                    let weight = distributions.sample(&mut thread_rng());
                    let connection = ConnectionGene::new(i, j + n_inputs, weight, true, (i - 1) * n_outputs + j);
                    genome.add_connection(connection);
                }
            }

            // bias -> output connections
            for i in 1..=n_outputs {
                let weight = distributions.sample(&mut thread_rng());
                let connection = ConnectionGene::new(n_inputs + n_outputs + 1, i + n_inputs, weight, true, n_inputs * n_outputs + i);
                genome.add_connection(connection);
            }

            Individual::new(genome)
        };

        let innovation = n_inputs * n_outputs + n_outputs;
        let nodes_nb = n_inputs + n_outputs + 1;

        let mut mutations = Vec::new();
        for i in 1..=n_inputs {
            for j in 1..=n_outputs {
                let connection = ConnectionGene::new(i, j + n_inputs, 0., true, (i - 1) * n_outputs + j); // weight does not matter here
                mutations.push((Mutation::NewConnection(connection), 0));
            }
        }

        for i in 1..=n_outputs {
            let connection = ConnectionGene::new(n_inputs + n_outputs + 1, i + n_inputs, 0., true, n_inputs * n_outputs + i); // weight does not matter here
            mutations.push((Mutation::NewConnection(connection), 0));
        }

        let history = History { innovation, nodes_nb, mutations, generation: 0, };
        let population = (0..population_size).map(|_| get_initial_individual(n_inputs, n_outputs, &weights_distribution)).collect::<Vec<_>>();

        (history, population)
    }

    fn next_generation(&mut self) {
        for individual in self.population.iter_mut() {
            individual.fitness = (self.config.evaluation_function)(individual);
        }

        self.history.generation += 1;

        let mut new_population = Vec::new();

        let mut sorted_population = self.population.clone();
        sorted_population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let survival_cutoff = (self.config.population_size as f32 * self.config.survival_threshold) as usize;
        let survivors = &sorted_population[..survival_cutoff];
        new_population.extend_from_slice(survivors);

        let mut rng = thread_rng();
        let weights_distribution = Normal::new(self.config.weights_mean, self.config.weights_stddev).unwrap();
        let perturbation_distribution = Normal::new(0., self.config.perturbation_stddev).unwrap();

        while new_population.len() < self.config.population_size as usize {
            let parent1 = survivors.choose(&mut rng).unwrap();
            let parent2 = survivors.choose(&mut rng).unwrap();

            let mut child = Individual::crossover(parent1, parent2);

            if rng.gen::<f32>() < self.config.connection_mutation_rate {
                child.mutate_add_connection(&mut self.history, &weights_distribution);
            }

            if rng.gen::<f32>() < self.config.node_mutation_rate {
                child.mutate_add_node(&mut self.history);
            }

            if rng.gen::<f32>() < self.config.weight_mutation_rate {
                child.mutate_weights(&weights_distribution, &perturbation_distribution);
            }

            new_population.push(child);
        }

        self.population = new_population;
    }

    pub fn run(&mut self) {
        for i in 1..=self.config.n_generations {
            self.next_generation();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover() {
        let node1 = NodeGene::new(1, NodeType::Input, SIGMOID);
        let node2 = NodeGene::new(2, NodeType::Input, SIGMOID);
        let node3 = NodeGene::new(3, NodeType::Input, SIGMOID);
        let node4 = NodeGene::new(4, NodeType::Output, SIGMOID);
        let node5 = NodeGene::new(5, NodeType::Hidden, SIGMOID);
        let node6 = NodeGene::new(6, NodeType::Hidden, SIGMOID);

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
        let node1 = NodeGene::new(1, NodeType::Input, SIGMOID);
        let node2 = NodeGene::new(2, NodeType::Output, SIGMOID);
        let connection = ConnectionGene::new(1, 2, 0.5, true, 1);
        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        genome.add_connection(connection.clone());
        let mut individual = Individual::new(genome);

        let mut history = History { innovation: 1, nodes_nb: 2, mutations: vec![(Mutation::NewConnection(connection), 0)], generation: 0};
        individual.mutate_add_node(&mut history);

        assert_eq!(individual.genome.nodes.len(), 3);
        assert_eq!(individual.genome.connections.len(), 3);
        assert_eq!(history.nodes_nb, 3);
        assert_eq!(history.innovation, 3);

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
        let node1 = NodeGene::new(1, NodeType::Input, SIGMOID);
        let node2 = NodeGene::new(2, NodeType::Output, SIGMOID);
        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        let mut individual = Individual::new(genome);

        let mut history = History { innovation: 0, nodes_nb: 2, mutations: vec![], generation: 0};
        let weights_distribution = Normal::new(0., 1.).unwrap();
        individual.mutate_add_connection(&mut history, &weights_distribution);

        assert_eq!(individual.genome.nodes.len(), 2);
        assert_eq!(individual.genome.connections.len(), 1);
        assert_eq!(history.innovation, 1);

        let new_connection = individual.genome.connections.iter().find(|c| c.innovation == 1).unwrap();
        assert_eq!(new_connection.in_node, 1);
        assert_eq!(new_connection.out_node, 2);
    }

    #[test]
    fn test_mutate_add_connection_already_connected_nodes() {
        let node1 = NodeGene::new(1, NodeType::Input, SIGMOID);
        let node2 = NodeGene::new(2, NodeType::Output, SIGMOID);
        let connection = ConnectionGene::new(1, 2, 0., true, 1);
        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        genome.add_connection(connection.clone());
        let mut individual = Individual::new(genome);

        let mut history = History { innovation: 1, nodes_nb: 2, mutations: vec![(Mutation::NewConnection(connection), 0)], generation: 0};
        let weights_distribution = Normal::new(0., 1.).unwrap();
        individual.mutate_add_connection(&mut history, &weights_distribution);

        assert_eq!(individual.genome.nodes.len(), 2);
        assert_eq!(individual.genome.connections.len(), 1);
        assert_eq!(history.innovation, 1);
    }

    #[test]
    fn test_network_conversion() {
        let node1 = NodeGene::new(1, NodeType::Input, IDENTITY);
        let node2 = NodeGene::new(2, NodeType::Input, IDENTITY);
        let node3 = NodeGene::new(3, NodeType::Output, IDENTITY);

        let conn_1_3 = ConnectionGene::new(1, 3, 0.5, true, 1);
        let conn_2_3 = ConnectionGene::new(2, 3, 0.5, true, 2);

        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        genome.add_node(node3);
        genome.add_connection(conn_1_3);
        genome.add_connection(conn_2_3);

        let individual = Individual::new(genome);
        let network = individual.to_neural_network();

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![1.]);
    }

    #[test]
    fn test_network_conversion_with_disabled_connection() {
        let node1 = NodeGene::new(1, NodeType::Input, IDENTITY);
        let node2 = NodeGene::new(2, NodeType::Input, IDENTITY);
        let node3 = NodeGene::new(3, NodeType::Output, IDENTITY);

        let conn_1_3 = ConnectionGene::new(1, 3, 0.5, true, 1);
        let conn_2_3 = ConnectionGene::new(2, 3, 0.5, false, 2);

        let mut genome = Genome::new();
        genome.add_node(node1);
        genome.add_node(node2);
        genome.add_node(node3);
        genome.add_connection(conn_1_3);
        genome.add_connection(conn_2_3);

        let individual = Individual::new(genome);
        let network = individual.to_neural_network();

        let inputs = vec![1., 1.];
        let outputs = network.feed_forward(&inputs);

        assert_eq!(outputs, vec![0.5]);
    }

    #[test]
    fn test_population_initialization() {
        let config = Config {
            population_size: 10,
            n_inputs: 3,
            n_outputs: 2,
            n_generations: 10,
            evaluation_function: |_: &Individual| 0.0,
            weights_mean: 0.0,
            weights_stddev: 1.0,
            perturbation_stddev: 1.,
            survival_threshold: 0.3,
            connection_mutation_rate: 0.1,
            node_mutation_rate: 0.1,
            weight_mutation_rate: 0.1,
        };

        let neat = Neat::new(config);

        assert_eq!(neat.population.len(), 10);
        assert_eq!(neat.history.innovation, 8);
        assert_eq!(neat.history.nodes_nb, 6);

        let individual = &neat.population[0];
        assert_eq!(individual.genome.nodes.len(), 6);
        assert_eq!(individual.genome.connections.len(), 8);

        let node_ids = individual.genome.nodes.iter().map(|n| n.id).collect::<Vec<_>>();
        assert_eq!(node_ids, vec![1, 2, 3, 4, 5, 6]);
        let innovation_ids = individual.genome.connections.iter().map(|c| c.innovation).collect::<Vec<_>>();
        assert_eq!(innovation_ids, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_activation() {
        let input_node = NodeGene::new(1, NodeType::Input, IDENTITY);
        let output_node = NodeGene::new(2, NodeType::Output, SIGMOID);
        let connection = ConnectionGene::new(1, 2, 1., true, 1);

        let mut genome = Genome::new();
        genome.add_node(input_node);
        genome.add_node(output_node);
        genome.add_connection(connection);

        let individual = Individual::new(genome);
        let network = individual.to_neural_network();

        let inputs = vec![0.];
        let outputs = network.feed_forward(&inputs);
        assert_eq!(outputs, vec![0.5]);
    }

    #[test]
    fn test_duplicate_add_connection_mutation() {
        let input_node = NodeGene::new(1, NodeType::Input, IDENTITY);
        let output_node = NodeGene::new(2, NodeType::Output, SIGMOID);

        let mut genome = Genome::new();
        genome.add_node(input_node);
        genome.add_node(output_node);

        let mut individual_1 = Individual::new(genome.clone());
        let mut individual_2 = Individual::new(genome);

        let mut history = History::new(2, 0, vec![]);
        let weights_distribution = Normal::new(0., 1.).unwrap();

        individual_1.mutate_add_connection(&mut history, &weights_distribution);
        individual_2.mutate_add_connection(&mut history, &weights_distribution);

        assert_eq!(history.innovation, 1);
    }

    #[test]
    fn test_duplicate_add_node_mutation() {
        let input_node = NodeGene::new(1, NodeType::Input, IDENTITY);
        let output_node = NodeGene::new(2, NodeType::Output, SIGMOID);
        let connection = ConnectionGene::new(1, 2, 1., true, 1);

        let mut genome = Genome::new();
        genome.add_node(input_node);
        genome.add_node(output_node);
        genome.add_connection(connection.clone());

        let mut individual_1 = Individual::new(genome.clone());
        let mut individual_2 = Individual::new(genome);

        let mut history = History::new(2, 1, vec![(Mutation::NewConnection(connection), 0)]);
        history.generation = 1;

        individual_1.mutate_add_node(&mut history);
        individual_2.mutate_add_node(&mut history);

        assert_eq!(history.innovation, 3);
    }
}
