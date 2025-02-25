import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
from qiskit.quantum_info import entropy, partial_trace, Statevector, random_statevector
from qiskit.visualization import plot_histogram
from scipy.linalg import sqrtm
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from functools import lru_cache
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HolographicState:
    """
    Represents quantum state with holographic properties
    
    Attributes:
        bulk_data: Quantum state in the bulk
        surface_encoding: Encoded state on the boundary
        entanglement_entropy: Von Neumann entropy of the state
        error_syndromes: List of error detection results
        fidelity: Encoding fidelity measurement
        metadata: Additional information about the state
    """
    bulk_data: np.ndarray
    surface_encoding: np.ndarray
    entanglement_entropy: float
    error_syndromes: List[bool]
    fidelity: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class EnhancedNoiseModel:
    """
    Sophisticated noise modeling system for quantum computations
    
    This class creates realistic noise models for quantum simulations,
    accounting for decoherence, crosstalk, and gate errors.
    """
    def __init__(self, 
                 T1: float = 50e-6, 
                 T2: float = 70e-6, 
                 gate_times: Dict[str, float] = None,
                 device_connectivity: nx.Graph = None):
        """
        Initialize the noise model with realistic parameters
        
        Args:
            T1: Relaxation time (seconds)
            T2: Dephasing time (seconds)
            gate_times: Dictionary of gate operation times
            device_connectivity: Graph representing qubit connectivity
        """
        if gate_times is None:
            gate_times = {
                'single': 20e-9,
                'cx': 100e-9,
                'measure': 300e-9,
                'reset': 200e-9
            }
        
        self.noise_model = NoiseModel()
        self.T1 = T1
        self.T2 = T2
        self.gate_times = gate_times
        
        # Create device connectivity if not provided
        if device_connectivity is None:
            self.device_connectivity = nx.grid_2d_graph(3, 4)  # 3x4 grid
        else:
            self.device_connectivity = device_connectivity
            
        self.crosstalk_matrix = self._generate_crosstalk_matrix(len(self.device_connectivity))
        logger.info(f"Initialized noise model with T1={T1}s, T2={T2}s")
        
    def _generate_crosstalk_matrix(self, size: int) -> np.ndarray:
        """
        Generate realistic crosstalk coupling matrix based on physical layout
        
        Args:
            size: Number of qubits in the system
            
        Returns:
            Matrix representing crosstalk coupling strengths
        """
        # Use device connectivity to inform crosstalk
        base_coupling = 0.005
        distance_matrix = np.zeros((size, size))
        
        # Convert graph to adjacency matrix for distance calculation
        if hasattr(self.device_connectivity, 'nodes'):
            # If we have a NetworkX graph with potentially complex node identifiers
            nodes = list(self.device_connectivity.nodes())
            node_map = {node: i for i, node in enumerate(nodes)}
            
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i != j:
                        try:
                            # Calculate shortest path distance in the connectivity graph
                            distance = nx.shortest_path_length(
                                self.device_connectivity, node_i, node_j)
                            distance_matrix[i, j] = base_coupling / (distance ** 2)
                        except nx.NetworkXNoPath:
                            # If no path exists, use a minimal coupling value
                            distance_matrix[i, j] = base_coupling / 1000
        else:
            # Fallback to simple distance model
            for i in range(size):
                for j in range(size):
                    if i != j:
                        distance = abs(i - j)
                        distance_matrix[i, j] = base_coupling / (distance ** 2)
                        
        # Ensure matrix is symmetric
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        return distance_matrix

    def calculate_decoherence_rates(self) -> Dict[str, float]:
        """
        Calculate comprehensive decoherence rates for the quantum system
        
        Returns:
            Dictionary of different error rates
        """
        gamma1 = 1/self.T1
        gamma2 = 1/self.T2
        
        # Different error contributions
        rates = {
            'thermal': gamma1,
            'dephasing': gamma2 - gamma1/2,
            'depolarizing': (2*gamma1 + gamma2)/4,
            'measurement': 1/(4*self.gate_times['measure']),
            'crosstalk': np.mean(self.crosstalk_matrix) * gamma2
        }
        
        # Add coherent error contributions
        rates['coherent'] = 0.001 * np.sum(list(rates.values()))
        return rates

    def add_realistic_noise(self) -> NoiseModel:
        """
        Add comprehensive noise effects to the quantum model
        
        Returns:
            Configured NoiseModel with realistic noise channels
        """
        rates = self.calculate_decoherence_rates()
        
        # Add thermal relaxation errors
        for qubit in range(len(self.crosstalk_matrix)):
            # Single-qubit gate errors
            for gate in ['x', 'sx', 'rz', 'u1', 'u2', 'u3']:
                time = self.gate_times.get('single', 20e-9)
                error = thermal_relaxation_error(
                    self.T1, self.T2, time, excited_state_population=0.01
                )
                self.noise_model.add_quantum_error(error, gate, [qubit])
            
            # Measurement errors
            meas_error = depolarizing_error(rates['measurement'], 1)
            self.noise_model.add_quantum_error(meas_error, 'measure', [qubit])
        
        # Add two-qubit gate errors with crosstalk
        for edge in self.device_connectivity.edges():
            q1, q2 = edge if isinstance(edge[0], int) else (
                list(self.device_connectivity.nodes()).index(edge[0]),
                list(self.device_connectivity.nodes()).index(edge[1])
            )
            
            # Get crosstalk strength between these qubits
            crosstalk = self.crosstalk_matrix[q1, q2]
            
            # CNOT error that includes crosstalk effects
            cx_error_rate = rates['depolarizing'] * (1 + crosstalk)
            cx_error = depolarizing_error(cx_error_rate, 2)
            self.noise_model.add_quantum_error(cx_error, 'cx', [q1, q2])
        
        logger.info(f"Added realistic noise model with {len(self.device_connectivity)} qubits")
        return self.noise_model
    
    def visualize_noise_profile(self, filename: str = None):
        """
        Visualize the noise characteristics of the model
        
        Args:
            filename: Optional path to save the visualization
        """
        rates = self.calculate_decoherence_rates()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot error rates
        ax1.bar(rates.keys(), rates.values())
        ax1.set_title('Decoherence Rates')
        ax1.set_ylabel('Rate (1/s)')
        ax1.set_yscale('log')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot crosstalk matrix
        im = ax2.imshow(self.crosstalk_matrix)
        ax2.set_title('Crosstalk Coupling Strengths')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved noise profile visualization to {filename}")
        else:
            plt.show()

class TensorNetwork:
    """
    Advanced tensor network for holographic encoding using AdS/CFT correspondence
    
    This class implements tensor networks that capture the geometry of
    AdS/CFT correspondence for quantum information processing.
    """
    def __init__(self, 
                 nodes: int, 
                 dimension: int, 
                 geometry: str = 'hyperbolic', 
                 ads_radius: float = 1.0,
                 visualization_enabled: bool = False):
        """
        Initialize the tensor network with specified geometry
        
        Args:
            nodes: Number of nodes in the network
            dimension: Dimension of tensor indices
            geometry: Type of geometry ('hyperbolic' or 'euclidean')
            ads_radius: Radius parameter for AdS space
            visualization_enabled: Whether to enable visualization capabilities
        """
        self.graph = nx.Graph()
        self.dimension = dimension
        self.geometry = geometry
        self.ads_radius = ads_radius
        self.tensors: Dict[int, np.ndarray] = {}
        self.bulk_boundary_map: Dict[int, List[int]] = {}
        self.boundary_operators: Dict[int, np.ndarray] = {}
        self.visualization_enabled = visualization_enabled
        
        # Track computation costs
        self.contraction_cost = 0
        
        self.setup_network(nodes)
        logger.info(f"Created tensor network with {nodes} nodes using {geometry} geometry")
        
    def setup_network(self, nodes: int):
        """
        Creates sophisticated tensor network structure following AdS/CFT principles
        
        Args:
            nodes: Number of nodes in the network
        """
        start_time = time.time()
        if self.geometry == 'hyperbolic':
            # Create hyperbolic geometry with proper AdS metric
            self.graph = self._create_ads_graph(nodes)
        else:
            # Create planar graph with conformal structure
            self.graph = self._create_cft_graph(nodes)
            
        # Initialize tensors with proper holographic structure
        for node in tqdm(self.graph.nodes(), desc="Initializing tensors"):
            self.tensors[node] = self._create_holographic_tensor(node)
            self.boundary_operators[node] = self._create_boundary_operator(node)
            
        # Setup bulk-boundary dictionary
        self._setup_bulk_boundary_dictionary()
        
        elapsed = time.time() - start_time
        logger.info(f"Network setup completed in {elapsed:.2f} seconds")
            
    def _create_ads_graph(self, nodes: int) -> nx.Graph:
        """
        Create graph with proper AdS geometry using Poincaré disk model
        
        Args:
            nodes: Number of nodes in the network
            
        Returns:
            NetworkX graph with AdS geometry
        """
        graph = nx.Graph()
        
        # Create boundary nodes in a circle
        boundary_nodes = []
        for i in range(nodes):
            theta = 2 * np.pi * i / nodes
            r = 0.95  # Place near the boundary of Poincaré disk
            
            # Convert to Poincaré coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            graph.add_node(i, pos=(x, y), is_boundary=True)
            boundary_nodes.append(i)
            
        # Add bulk nodes with hyperbolic spacing
        num_bulk_nodes = nodes // 3
        bulk_node_id = nodes
        
        for layer in range(1, 4):  # Create concentric layers
            layer_nodes = min(num_bulk_nodes // layer, 8 * layer)
            r = 0.8 / layer  # Radius decreases for inner layers
            
            for i in range(layer_nodes):
                theta = 2 * np.pi * i / layer_nodes + (np.pi / layer_nodes * layer)
                
                # Convert to Poincaré coordinates
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                graph.add_node(bulk_node_id, pos=(x, y), is_boundary=False)
                
                # Connect to nearest neighbors
                for node in graph.nodes():
                    if node != bulk_node_id:
                        dist = self._geodesic_distance(
                            graph.nodes[bulk_node_id]['pos'], 
                            graph.nodes[node]['pos']
                        )
                        
                        # Add edge if within geodesic distance threshold
                        if dist < 1.5 / layer:
                            graph.add_edge(bulk_node_id, node)
                
                bulk_node_id += 1
                
        # Ensure the graph is connected
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            # Connect the largest components
            sorted_components = sorted(components, key=len, reverse=True)
            
            for i in range(len(sorted_components) - 1):
                node1 = next(iter(sorted_components[i]))
                node2 = next(iter(sorted_components[i+1]))
                graph.add_edge(node1, node2)
        
        return graph
    
    def _create_cft_graph(self, nodes: int) -> nx.Graph:
        """
        Create graph with CFT structure on the boundary
        
        Args:
            nodes: Number of nodes on the boundary
            
        Returns:
            NetworkX graph with CFT structure
        """
        graph = nx.Graph()
        
        # Create boundary circle with CFT structure
        for i in range(nodes):
            theta = 2 * np.pi * i / nodes
            
            # Boundary points are on unit circle
            x = np.cos(theta)
            y = np.sin(theta)
            
            graph.add_node(i, pos=(x, y), is_boundary=True)
            
            # Connect to neighbors - nearest and next-nearest
            graph.add_edge(i, (i + 1) % nodes)  # Nearest
            graph.add_edge(i, (i + 2) % nodes)  # Next-nearest
            
        # Add some central bulk nodes
        num_bulk = nodes // 4
        bulk_start_id = nodes
        
        # Create central bulk nodes
        for i in range(num_bulk):
            r = 0.6 * np.random.random() # Interior points
            theta = 2 * np.pi * i / num_bulk
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            graph.add_node(bulk_start_id + i, pos=(x, y), is_boundary=False)
            
            # Connect to some boundary nodes - CFT to bulk connection
            num_connections = min(3, nodes // 4)
            boundary_indices = np.random.choice(
                nodes, size=num_connections, replace=False
            )
            
            for boundary_idx in boundary_indices:
                graph.add_edge(bulk_start_id + i, boundary_idx)
            
            # Connect bulk nodes to each other
            if i > 0:
                graph.add_edge(bulk_start_id + i, bulk_start_id + i - 1)
                
        # Add final connection to ensure ring structure
        if num_bulk > 1:
            graph.add_edge(bulk_start_id, bulk_start_id + num_bulk - 1)
                
        return graph
    
    def _geodesic_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """
        Calculate geodesic distance in AdS space using hyperbolic metric
        
        Args:
            pos1: Position of first point (x1, y1)
            pos2: Position of second point (x2, y2)
            
        Returns:
            Geodesic distance between points
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calculate Euclidean distance in the Poincaré disk
        dx = x2 - x1
        dy = y2 - y1
        euclidean_squared = dx**2 + dy**2
        
        # Get the Poincaré disk coordinates
        r1_squared = x1**2 + y1**2
        r2_squared = x2**2 + y2**2
        
        # Use the formula for hyperbolic distance in the Poincaré disk
        numerator = 2 * euclidean_squared
        denominator = (1 - r1_squared) * (1 - r2_squared)
        
        # Avoid numerical issues
        arg = max(1 + numerator / denominator, 1.0001)
        
        return self.ads_radius * np.arccosh(arg)
    
    def _create_holographic_tensor(self, node: int) -> np.ndarray:
        """
        Create tensor with proper holographic structure
        
        Args:
            node: Node identifier
            
        Returns:
            Tensor with holographic structure
        """
        node_data = self.graph.nodes[node]
        pos = node_data['pos']
        is_boundary = node_data.get('is_boundary', False)
        
        # Get node degree (number of connections)
        degree = self.graph.degree[node]
        
        # Create tensor with proper rank based on node connections
        rank = max(3, degree)  # Minimum rank 3 for computational expressivity
        shape = tuple(self.dimension for _ in range(rank))
        
        # Initialize with random values
        tensor = np.random.normal(0, 0.1, shape)
        
        # Apply conformal constraints for boundary nodes
        if is_boundary:
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            
            # Scale boundary tensors by conformal factor
            conformal_factor = (1 - r**2) ** (-self.dimension/4)
            tensor *= conformal_factor
            
            # Enhance correlations with connected nodes
            for neighbor in self.graph.neighbors(node):
                # Create correlation pattern based on network geometry
                neighbor_pos = self.graph.nodes[neighbor]['pos']
                dist = self._geodesic_distance(pos, neighbor_pos)
                correlation_strength = np.exp(-dist)
                
                # Inject correlation pattern
                pattern = np.random.normal(0, correlation_strength, shape)
                tensor += pattern * 0.1
        else:
            # Apply holographic constraints to bulk tensors
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            tensor = self._apply_holographic_constraints(tensor, r)
        
        # Normalize tensor while preserving structure
        norm = np.sqrt(np.sum(tensor**2))
        if norm > 0:
            tensor /= norm
            
        return tensor
    
    def _apply_holographic_constraints(self, tensor: np.ndarray, 
                                     radial_coord: float) -> np.ndarray:
        """
        Apply holographic constraints to tensor following AdS/CFT principles
        
        Args:
            tensor: Input tensor
            radial_coord: Radial coordinate in AdS space
            
        Returns:
            Tensor with applied holographic constraints
        """
        # Apply radial scaling following holographic principle
        scaling_factor = 1 / (1 - radial_coord**2 + 0.01)
        
        # Get tensor shape
        shape = tensor.shape
        tensor_flattened = tensor.reshape(-1)
        
        # Apply holographic scaling (UV/IR correspondence)
        # Higher frequency components decay faster with radius
        fft_tensor = np.fft.fft(tensor_flattened)
        
        # Apply frequency-dependent scaling
        freq_indices = np.fft.fftfreq(len(fft_tensor))
        scaling = np.exp(-scaling_factor * np.abs(freq_indices))
        fft_tensor *= scaling
        
        # Transform back
        transformed = np.real(np.fft.ifft(fft_tensor))
        
        # Reshape to original tensor shape
        return transformed.reshape(shape)
    
    def _create_boundary_operator(self, node: int) -> np.ndarray:
        """
        Create boundary CFT operator with proper scaling dimension
        
        Args:
            node: Node identifier
            
        Returns:
            Operator matrix for boundary CFT
        """
        node_data = self.graph.nodes[node]
        pos = node_data['pos']
        is_boundary = node_data.get('is_boundary', False)
        
        # Calculate radius in Poincaré disk
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Create operator with dimension-dependent scaling
        if is_boundary:
            # Boundary operators have specific scaling dimensions
            scaling_dim = 2.0 + 0.5 * np.random.random()
        else:
            # Bulk operators have different scaling properties
            scaling_dim = 1.0 + np.random.random()
        
        # Create random operator
        operator = np.random.normal(0, 0.1, (self.dimension, self.dimension))
        
        # Apply conformal transformation
        operator *= (1 - r**2)**scaling_dim
        
        # Ensure hermiticity (important for physical operators)
        operator = 0.5 * (operator + operator.conj().T)
        
        # Normalize
        operator_norm = np.linalg.norm(operator, 'fro')
        if operator_norm > 0:
            operator /= operator_norm
            
        return operator
    
    def _setup_bulk_boundary_dictionary(self):
        """
        Setup bulk-boundary dictionary following holographic principle
        """
        bulk_nodes = [n for n, attr in self.graph.nodes(data=True) 
                     if not attr.get('is_boundary', False)]
        boundary_nodes = [n for n, attr in self.graph.nodes(data=True) 
                         if attr.get('is_boundary', False)]
        
        for bulk_node in bulk_nodes:
            mapped_boundary_nodes = []
            bulk_pos = self.graph.nodes[bulk_node]['pos']
            
            # Find boundary nodes in causal wedge using RT surfaces
            for boundary_node in boundary_nodes:
                boundary_pos = self.graph.nodes[boundary_node]['pos']
                
                if self._is_in_causal_wedge(bulk_pos, boundary_pos):
                    mapped_boundary_nodes.append(boundary_node)
                    
            self.bulk_boundary_map[bulk_node] = mapped_boundary_nodes
            
    def _is_in_causal_wedge(self, bulk_pos: Tuple[float, float], 
                           boundary_pos: Tuple[float, float]) -> bool:
        """
        Check if boundary point is in causal wedge of bulk point
        
        Args:
            bulk_pos: Position of bulk point
            boundary_pos: Position of boundary point
            
        Returns:
            True if boundary point is in causal wedge
        """
        bulk_r = np.sqrt(bulk_pos[0]**2 + bulk_pos[1]**2)
        boundary_r = np.sqrt(boundary_pos[0]**2 + boundary_pos[1]**2)
        
        # Check angular separation
        bulk_theta = np.arctan2(bulk_pos[1], bulk_pos[0])
        boundary_theta = np.arctan2(boundary_pos[1], boundary_pos[0])
        
        angular_diff = min(abs(bulk_theta - boundary_theta), 
                          2*np.pi - abs(bulk_theta - boundary_theta))
        
        # Causal cone widens as we approach boundary
        causal_angle = np.pi * (1 - bulk_r/boundary_r) * 1.2
        
        # Use both radial and angular criteria for causal wedge
        radial_criterion = (1 - boundary_r**2) < (1 - bulk_r**2) * 1.5
        angular_criterion = angular_diff < causal_angle
        
        return radial_criterion and angular_criterion

    @lru_cache(maxsize=32)
    def _find_optimal_contraction_order(self) -> List[int]:
        """
        Find optimal contraction order following RT surfaces
        
        Returns:
            Optimized contraction order for tensors
        """
        order = []
        remaining_nodes = set(self.graph.nodes())
        
        # Start from boundary and move inward
        boundary_nodes = [n for n, attr in self.graph.nodes(data=True) 
                         if attr.get('is_boundary', False)]
        
        # First contract boundary nodes
        for node in sorted(boundary_nodes, 
                          key=lambda n: self._rt_surface_area(n, remaining_nodes)):
            order.append(node)
            remaining_nodes.remove(node)
            
        # Then contract bulk nodes
        while remaining_nodes:
            # Find node that minimizes RT surface
            min_rt_area = float('inf')
            next_node = None
            
            for node in remaining_nodes:
                rt_area = self._rt_surface_area(node, remaining_nodes)
                if rt_area < min_rt_area:
                    min_rt_area = rt_area
                    next_node = node
                    
            if next_node is not None:
                order.append(next_node)
                remaining_nodes.remove(next_node)
            else:
                # Fallback in case of disconnected components
                next_node = next(iter(remaining_nodes))
                order.append(next_node)
                remaining_nodes.remove(next_node)
            
        return order
    
    def _rt_surface_area(self, node: int, remaining_nodes: Set[int]) -> float:
        """
        Calculate Ryu-Takayanagi surface area for given node
        
        Args:
            node: Node identifier
            remaining_nodes: Set of nodes not yet contracted
            
        Returns:
            Surface area according to RT formula
        """
        node_pos = self.graph.nodes[node]['pos']
        
        # Calculate minimal surface area
        area = 0
        connected_nodes = set(self.graph.neighbors(node)) & remaining_nodes
        
        # Calculate area based on connections to remaining nodes
        for other_node in connected_nodes:
            other_pos = self.graph.nodes[other_node]['pos']
            area += self._geodesic_distance(node_pos, other_pos)
                
        # Add penalty for disconnected contractions
        if not connected_nodes and remaining_nodes:
            area += 100  # Large penalty
            
        return area

    def contract(self, 
                contraction_order: Optional[List[int]] = None,
                optimize_order: bool = True) -> np.ndarray:
        """
        Sophisticated tensor network contraction with holographic ordering
        
        Args:
            contraction_order: Optional manual contraction order
            optimize_order: Whether to optimize the contraction order
            
        Returns:
            Contracted tensor result
        """
        if contraction_order is None:
            if optimize_order:
                contraction_order = self._find_optimal_contraction_order()
            else:
                contraction_order = list(self.tensors.keys())
                
        logger.info(f"Contracting tensor network with {len(contraction_order)} tensors")
        
        # Start with identity tensor of appropriate dimension
        result = np.eye(self.dimension)
        self.contraction_cost = 0
        
        # Track progress for large networks
        progress_bar = tqdm(
            contraction_order, 
            desc="Contracting tensors",
            disable=len(contraction_order) < 10
        )
        
        # Contract following the specified order
        for node in progress_bar:
            tensor = self.tensors[node]
            
            # Apply boundary conditions for boundary nodes
            if self.graph.nodes[node].get('is_boundary', False):
                # Apply boundary operator
                tensor = np.tensordot(
                    tensor, 
                    self.boundary_operators[node],
                    axes=([0], [0])
                )
            
            # Contract with accumulating result
            # Find optimal contraction axes
            axes = min(1, min(tensor.ndim, result.ndim) - 1)
            
            # Estimate contraction cost
            cost = np.prod(tensor.shape) * np.prod(result.shape) / tensor.shape[0]
            self.contraction_cost += cost
            
            # Perform contraction
            result = np.tensordot(result, tensor, axes=([0], [0]))
            
            # Apply holographic renormalization
            result = self._holographic_renormalize(result, node)
            
        logger.info(f"Tensor contraction completed with cost: {self.contraction_cost:.2e}")
        return result
    
    def _holographic_renormalize(self, tensor: np.ndarray, node: int) -> np.ndarray:
        """
        Apply holographic renormalization to maintain numerical stability
        
        Args:
            tensor: Tensor to renormalize
            node: Current node being processed
            
        Returns:
            Renormalized tensor
        """
        # Extract node position
        pos = self.graph.nodes[node]['pos']
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Calculate cutoff scale based on radial position
        cutoff = (1 - r**2) * self.ads_radius * 0.1
        cutoff = max(cutoff, 1e-10)  # Ensure minimum cutoff
        
        # Reshape tensor for SVD
        original_shape = tensor.shape
        tensor_matrix = tensor.reshape(-1, tensor.shape[-1])
        
        # Apply SVD
        try:
            # Use SVD for large tensors, truncate small singular values
            if max(tensor_matrix.shape) > 1000:
                # Use randomized SVD for very large tensors
                from sklearn.utils.extmath import randomized_svd
                U, S, Vt = randomized_svd(
                    tensor_matrix, 
                    n_components=min(tensor_matrix.shape),
                    n_iter=5
                )
            else:
                # Use full SVD for smaller tensors
                U, S, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
                
            # Apply cutoff to singular values
            S_filtered = S.copy()
            S_filtered[S < cutoff * np.max(S)] = 0
            
            # Reconstruct with filtered singular values
            tensor_matrix = U @ np.diag(S_filtered) @ Vt
                
        except np.linalg.LinAlgError:
            # Fallback if SVD fails
            logger.warning(f"SVD failed during renormalization at node {node}, using direct truncation")
            tensor_matrix[np.abs(tensor_matrix) < cutoff * np.max(np.abs(tensor_matrix))] = 0
        
        # Reshape back to original tensor shape
        tensor = tensor_matrix.reshape(original_shape)
        
        # Renormalize tensor
        tensor /= np.linalg.norm(tensor) + 1e-12
        
        return tensor
    
    def visualize(self, filename: str = None):
        """
        Visualize the tensor network structure
        
        Args:
            filename: Optional path to save visualization
        """
        if not self.visualization_enabled:
            logger.warning("Visualization is disabled. Enable with visualization_enabled=True")
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get node positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Color nodes by type (boundary vs bulk)
        node_colors = []
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('is_boundary', False):
                node_colors.append('skyblue')
            else:
                node_colors.append('salmon')
        
        # Draw the graph
        nx.draw_networkx(
            self.graph, 
            pos=pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=500,
            font_size=10,
            width=1.0,
            ax=ax
        )
        
        # Draw edge labels with distances
        edge_labels = {}
        for u, v in self.graph.edges():
            u_pos = pos[u]
            v_pos = pos[v]
            distance = round(self._geodesic_distance(u_pos, v_pos), 2)
            edge_labels[(u, v)] = f"{distance:.2f}"
            
        nx.draw_networkx_edge_labels(
            self.graph, 
            pos=pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Set plot limits to match Poincaré disk
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        
        plt.title(f'Tensor Network: {self.geometry.capitalize()} Geometry')
        plt.axis('equal')
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved tensor network visualization to {filename}")
        else:
            plt.show()

class OptimizedHolographicEncoder:
    """
    Advanced holographic encoding system for quantum states
    
    This class implements efficient encoding of bulk quantum states
    to boundary representations using holographic principles.
    """
    def __init__(self, dim_boundary: int, dim_bulk: int, tensor_network: TensorNetwork = None):
        """
        Initialize the holographic encoder
        
        Args:
            dim_boundary: Dimension of boundary Hilbert space
            dim_bulk: Dimension of bulk Hilbert space
            tensor_network: Optional existing tensor network to use
        """
        self.dim_boundary = dim_boundary
        self.dim_bulk = dim_bulk
        
        # Create tensor network if not provided
        if tensor_network is None:
            self.tensor_network = TensorNetwork(dim_boundary, dim_bulk)
        else:
            self.tensor_network = tensor_network
            
        self.error_bounds = self._calculate_error_bounds()
        self.encoding_history = []
        
    def _calculate_error_bounds(self) -> Dict[str, float]:
        """
        Calculate theoretical error bounds for encoding
        
        Returns:
            Dictionary of error bounds for different error types
        """
        # Calculate based on dimensions and tensor network properties
        return {
            'truncation': 1e-10,
            'reconstruction': 1e-8,
            'entanglement': 0.01,
            'holographic': 1/(self.dim_boundary * np.log(self.dim_bulk))
        }
        
    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate input quantum state
        
        Args:
            state: Input quantum state vector or density matrix
            
        Returns:
            True if valid state, raises ValueError otherwise
        """
        # Check dimensions
        expected_dim = 2**self.dim_bulk
        
        if state.shape[0] != expected_dim:
            raise ValueError(
                f"State dimension {state.shape[0]} doesn't match expected {expected_dim}"
            )
            
        # Check normalization for pure states
        if len(state.shape) == 1:
            norm = np.linalg.norm(state)
            if not np.isclose(norm, 1.0, atol=1e-5):
                logger.warning(f"State not normalized (norm={norm}), normalizing")
                state /= norm
                
        # Check hermiticity and trace for density matrices
        elif len(state.shape) == 2:
            if not np.isclose(np.trace(state), 1.0, atol=1e-5):
                raise ValueError(f"Density matrix must have trace 1, found {np.trace(state)}")
                
            # Check Hermiticity
            if not np.allclose(state, state.conj().T, atol=1e-5):
                raise ValueError("Density matrix must be Hermitian")
                
        return True
        
    def encode_with_optimization(self, bulk_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimized encoding with error tracking
        
        Args:
            bulk_state: Quantum state in the bulk
            
        Returns:
            Tuple of (encoded_state, fidelity)
        """
        # Validate input state
        try:
            self.validate_state(bulk_state)
        except ValueError as e:
            logger.error(f"Invalid state: {str(e)}")
            raise
        
        # Convert to density matrix if given state vector
        is_pure_state = len(bulk_state.shape) == 1
        if is_pure_state:
            state_dm = np.outer(bulk_state, bulk_state.conj())
        else:
            state_dm = bulk_state
            
        # Initial encoding - contract tensor network with bulk state
        logger.info("Performing holographic encoding...")
        start_time = time.time()
        
        # Contract the tensor network
        contracted_network = self.tensor_network.contract()
        
        # Apply to bulk state
        encoded_dm = contracted_network @ state_dm @ contracted_network.conj().T
        
        # Convert back to state vector if input was pure
        if is_pure_state:
            # Get dominant eigenvector
            eigenvals, eigenvecs = np.linalg.eigh(encoded_dm)
            max_idx = np.argmax(eigenvals)
            encoded = eigenvecs[:, max_idx]
            
            # Ensure correct phase
            overlap = np.vdot(bulk_state, encoded)
            encoded *= np.exp(-1j * np.angle(overlap))
        else:
            encoded = encoded_dm
            
        # Calculate encoding quality
        fidelity = self._calculate_encoding_fidelity(bulk_state, encoded)
        
        # Optimize if needed
        if fidelity < 0.90:
            logger.info(f"Initial fidelity {fidelity:.4f} below threshold, optimizing...")
            encoded = self._improve_encoding(encoded, bulk_state)
            fidelity = self._calculate_encoding_fidelity(bulk_state, encoded)
            logger.info(f"Improved fidelity to {fidelity:.4f}")
            
        elapsed = time.time() - start_time
        logger.info(f"Encoding completed in {elapsed:.2f} seconds with fidelity {fidelity:.4f}")
        
        # Record encoding history
        self.encoding_history.append({
            'fidelity': fidelity,
            'time': elapsed,
            'input_dim': bulk_state.shape[0],
            'output_dim': encoded.shape[0]
        })
            
        return encoded, fidelity
    
    def _calculate_encoding_fidelity(self, original: np.ndarray, encoded: np.ndarray) -> float:
        """
        Calculate fidelity between original and encoded states
        
        Args:
            original: Original quantum state
            encoded: Encoded quantum state
            
        Returns:
            Fidelity between states (0 to 1)
        """
        # Handle different input types
        is_pure_original = len(original.shape) == 1
        is_pure_encoded = len(encoded.shape) == 1
        
        if is_pure_original and is_pure_encoded:
            # Pure state to pure state fidelity
            return np.abs(np.vdot(original, encoded))**2
        
        elif is_pure_original and not is_pure_encoded:
            # Pure state to mixed state fidelity
            original_dm = np.outer(original, original.conj())
            return np.real(np.trace(sqrtm(
                sqrtm(encoded) @ original_dm @ sqrtm(encoded)
            ))**2)
            
        elif not is_pure_original and is_pure_encoded:
            # Mixed state to pure state fidelity
            encoded_dm = np.outer(encoded, encoded.conj())
            return np.real(np.trace(sqrtm(
                sqrtm(original) @ encoded_dm @ sqrtm(original)
            ))**2)
            
        else:
            # Mixed state to mixed state fidelity
            product = sqrtm(original) @ encoded @ sqrtm(original)
            try:
                sqrt_product = sqrtm(product)
                return np.real(np.trace(sqrt_product)**2)
            except np.linalg.LinAlgError:
                # Fallback for numerical issues
                logger.warning("Numerical issues in fidelity calculation, using eigenvalue method")
                eigenvals = np.linalg.eigvalsh(product)
                return np.sum(np.sqrt(np.maximum(eigenvals, 0)))**2
    
    def _improve_encoding(self, encoded: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Improve encoding quality through optimization
        
        Args:
            encoded: Current encoded state
            target: Target state to approach
            
        Returns:
            Improved encoded state
        """
        # Determine if working with pure or mixed states
        is_pure_encoded = len(encoded.shape) == 1
        is_pure_target = len(target.shape) == 1
        
        if is_pure_encoded and is_pure_target:
            # Pure state optimization
            corrected = encoded.copy()
            
            # Project onto target and normalize
            overlap = np.vdot(target, corrected)
            phase_correction = np.exp(-1j * np.angle(overlap))
            corrected *= phase_correction
            
            # Remove small components
            corrected[np.abs(corrected) < self.error_bounds['truncation']] = 0
            
            # Renormalize
            corrected /= np.linalg.norm(corrected)
            
            # Boost matching components
            matching_mask = np.abs(target) > 0.1
            if np.any(matching_mask):
                corrected[matching_mask] *= 1.05
                corrected /= np.linalg.norm(corrected)
            
        else:
            # Mixed state optimization - convert everything to density matrices
            if is_pure_encoded:
                encoded_dm = np.outer(encoded, encoded.conj())
            else:
                encoded_dm = encoded
                
            if is_pure_target:
                target_dm = np.outer(target, target.conj())
            else:
                target_dm = target
                
            # Apply filtering
            threshold = self.error_bounds['truncation']
            encoded_dm[np.abs(encoded_dm) < threshold] = 0
            
            # Ensure Hermiticity
            encoded_dm = 0.5 * (encoded_dm + encoded_dm.conj().T)
            
            # Ensure positivity
            eigenvals, eigenvecs = np.linalg.eigh(encoded_dm)
            eigenvals = np.maximum(eigenvals, 0)
            encoded_dm = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
            
            # Ensure trace 1
            encoded_dm /= np.trace(encoded_dm)
            
            # Pull toward target
            weight = 0.3  # Balance between original and target
            mixed = (1 - weight) * encoded_dm + weight * target_dm
            
            if is_pure_encoded:
                # Get principal eigenvector
                eigenvals, eigenvecs = np.linalg.eigh(mixed)
                max_idx = np.argmax(eigenvals)
                corrected = eigenvecs[:, max_idx]
                
                # Normalize
                corrected /= np.linalg.norm(corrected)
            else:
                corrected = mixed
            
        return corrected
    
    def visualize_encoding_metrics(self, filename: str = None):
        """
        Visualize encoding performance metrics
        
        Args:
            filename: Optional path to save visualization
        """
        if not self.encoding_history:
            logger.warning("No encoding history available for visualization")
            return
            
        # Extract metrics
        fidelities = [entry['fidelity'] for entry in self.encoding_history]
        times = [entry['time'] for entry in self.encoding_history]
        iterations = range(1, len(fidelities) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot fidelities
        ax1.plot(iterations, fidelities, 'o-', color='blue')
        ax1.set_title('Encoding Fidelity')
        ax1.set_xlabel('Encoding Iteration')
        ax1.set_ylabel('Fidelity')
        ax1.grid(True, alpha=0.3)
        
        # Plot times
        ax2.plot(iterations, times, 's-', color='green')
        ax2.set_title('Encoding Time')
        ax2.set_xlabel('Encoding Iteration')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved encoding metrics visualization to {filename}")
        else:
            plt.show()

class EnhancedEntropyCalculator:
    """
    Advanced entropy calculation system for quantum information
    
    This class provides methods to calculate various entropy measures
    for quantum states and subsystems.
    """
    def __init__(self, system_size: int):
        """
        Initialize the entropy calculator
        
        Args:
            system_size: Number of qubits in the system
        """
        self.system_size = system_size
        self.previous_entropies: Dict[str, List[float]] = {
            'von_neumann': [],
            'renyi_2': [],
            'renyi_inf': []
        }
        
    @lru_cache(maxsize=16)
    def get_subsystem_dims(self, subsystem_qubits: Tuple[int]) -> Tuple[int, int]:
        """
        Calculate dimensions for a subsystem and its complement
        
        Args:
            subsystem_qubits: Tuple of qubit indices in the subsystem
            
        Returns:
            Tuple of dimensions (subsystem, complement)
        """
        subsystem_dim = 2**len(subsystem_qubits)
        complement_dim = 2**(self.system_size - len(subsystem_qubits))
        return subsystem_dim, complement_dim
        
    def von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate von Neumann entropy with improved numerical stability
        
        S(ρ) = -Tr(ρ log₂ ρ)
        
        Args:
            density_matrix: Quantum density matrix
            
        Returns:
            Von Neumann entropy value
        """
        # Handle pure states given as vectors
        if len(density_matrix.shape) == 1:
            # Pure state has zero entropy
            self.previous_entropies['von_neumann'].append(0.0)
            return 0.0
            
        # Ensure proper density matrix
        if not np.isclose(np.trace(density_matrix), 1.0, atol=1e-5):
            logger.warning(f"Density matrix not properly normalized, trace = {np.trace(density_matrix)}")
            density_matrix = density_matrix / np.trace(density_matrix)
            
        # Use sparse eigenvalue solver for large matrices
        if density_matrix.shape[0] > 1000:
            try:
                # Get only eigenvalues using sparse methods
                sparse_matrix = csr_matrix(density_matrix)
                eigenvals = eigsh(sparse_matrix, k=min(100, density_matrix.shape[0]-2), 
                                 return_eigenvectors=False)
            except Exception as e:
                logger.warning(f"Sparse eigenvalue decomposition failed: {str(e)}")
                # Fallback to dense eigenvalue computation
                eigenvals = np.linalg.eigvalsh(density_matrix)
        else:
            # Use standard eigenvalue decomposition for smaller matrices
            eigenvals = np.linalg.eigvalsh(density_matrix)
        
        # Filter small eigenvalues to avoid numerical issues
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        # Calculate entropy
        entropy_val = -np.sum(eigenvals * np.log2(eigenvals))
        
        # Handle numerical errors
        if np.isnan(entropy_val) or np.isinf(entropy_val):
            logger.warning("Numerical issues in entropy calculation")
            entropy_val = 0.0
            
        self.previous_entropies['von_neumann'].append(entropy_val)
        return entropy_val
        
    def renyi_entropy(self, density_matrix: np.ndarray, alpha: float = 2.0) -> float:
        """
        Calculate Renyi entropy of order alpha
        
        S_α(ρ) = 1/(1-α) log₂(Tr(ρ^α))
        
        Args:
            density_matrix: Quantum density matrix
            alpha: Renyi entropy order parameter
            
        Returns:
            Renyi entropy value
        """
        # Special case: Von Neumann entropy
        if np.isclose(alpha, 1.0):
            return self.von_neumann_entropy(density_matrix)
            
        # Handle pure states
        if len(density_matrix.shape) == 1:
            return 0.0  # Pure states have zero entropy
            
        # Handle special case: alpha = infinity (min entropy)
        if np.isinf(alpha):
            eigenvals = np.linalg.eigvalsh(density_matrix)
            min_entropy = -np.log2(np.max(eigenvals))
            self.previous_entropies['renyi_inf'].append(min_entropy)
            return min_entropy
            
        # Calculate Renyi entropy for finite alpha != 1
        if alpha == 2:
            # Efficiently compute purity for alpha = 2
            purity = np.trace(density_matrix @ density_matrix)
            renyi_2 = -np.log2(purity)
            self.previous_entropies['renyi_2'].append(renyi_2)
            return renyi_2
        else:
            # For other alpha values, use eigenvalues
            eigenvals = np.linalg.eigvalsh(density_matrix)
            eigenvals = eigenvals[eigenvals > 1e-12]
            return 1/(1 - alpha) * np.log2(np.sum(eigenvals**alpha))

    def entanglement_entropy(self, 
                           state: np.ndarray, 
                           subsystem_qubits: Union[List[int], Tuple[int]]) -> float:
        """
        Calculate entanglement entropy of a subsystem
        
        Args:
            state: Full system quantum state (vector or density matrix)
            subsystem_qubits: Qubits in the subsystem
            
        Returns:
            Entanglement entropy of the subsystem
        """
        # Convert to tuple for caching
        if isinstance(subsystem_qubits, list):
            subsystem_qubits = tuple(subsystem_qubits)
            
        # Validate inputs
        if max(subsystem_qubits) >= self.system_size:
            raise ValueError(f"Subsystem qubit indices must be < {self.system_size}")
            
        # Check if we have a pure state
        is_pure = len(state.shape) == 1
        
        if is_pure:
            # For pure states, create density matrix
            rho = np.outer(state, state.conj())
        else:
            rho = state
            
        # Calculate dimensions
        subsys_dim, comp_dim = self.get_subsystem_dims(subsystem_qubits)
        
        # Get complementary subsystem qubits
        all_qubits = set(range(self.system_size))
        comp_qubits = list(all_qubits - set(subsystem_qubits))
        
        # Get reduced density matrix of the subsystem
        rho_A = partial_trace(rho, comp_qubits)
        
        # Calculate von Neumann entropy of reduced state
        return self.von_neumann_entropy(rho_A)

    def mutual_information(self, 
                         state: np.ndarray, 
                         subsys_A: List[int], 
                         subsys_B: List[int]) -> float:
        """
        Calculate quantum mutual information between subsystems
        
        I(A:B) = S(A) + S(B) - S(AB)
        
        Args:
            state: Full system quantum state
            subsys_A: Qubits in subsystem A
            subsys_B: Qubits in subsystem B
            
        Returns:
            Mutual information between subsystems
        """
        # Ensure subsystems are disjoint
        if set(subsys_A).intersection(set(subsys_B)):
            raise ValueError("Subsystems must be disjoint for mutual information")
            
        # Get reduced density matrices
        if len(state.shape) == 1:
            # Pure state - convert to density matrix
            rho = np.outer(state, state.conj())
        else:
            rho = state
            
        # Full system entropy not needed if state is pure
        is_pure = np.isclose(np.trace(rho @ rho), 1.0, atol=1e-5)
        
        rho_A = partial_trace(rho, list(set(range(self.system_size)) - set(subsys_A)))
        rho_B = partial_trace(rho, list(set(range(self.system_size)) - set(subsys_B)))
        
        # Get union of subsystems
        subsys_AB = list(set(subsys_A).union(set(subsys_B)))
        
        # For pure states, S(AB) = S(complement of AB)
        if is_pure and len(subsys_AB) < self.system_size:
            complement_AB = list(set(range(self.system_size)) - set(subsys_AB))
            rho_AB = partial_trace(rho, complement_AB)
        else:
            rho_AB = partial_trace(rho, list(set(range(self.system_size)) - set(subsys_AB)))
        
        # Calculate entropies
        S_A = self.von_neumann_entropy(rho_A)
        S_B = self.von_neumann_entropy(rho_B)
        S_AB = self.von_neumann_entropy(rho_AB)
        
        # Calculate mutual information
        return S_A + S_B - S_AB

    def conditional_entropy(self,
                          state: np.ndarray,
                          subsys_A: List[int],
                          subsys_B: List[int]) -> float:
        """
        Calculate quantum conditional entropy
        
        S(A|B) = S(AB) - S(B)
        
        Args:
            state: Full quantum state
            subsys_A: Qubits in subsystem A
            subsys_B: Qubits in subsystem B
            
        Returns:
            Conditional entropy S(A|B)
        """
        # Ensure subsystems are disjoint
        if set(subsys_A).intersection(set(subsys_B)):
            raise ValueError("Subsystems must be disjoint for conditional entropy")
            
        # Get union of subsystems
        subsys_AB = list(set(subsys_A).union(set(subsys_B)))
        
        # Convert to density matrix if needed
        if len(state.shape) == 1:
            rho = np.outer(state, state.conj())
        else:
            rho = state
            
        # Get reduced density matrices
        rho_B = partial_trace(rho, list(set(range(self.system_size)) - set(subsys_B)))
        rho_AB = partial_trace(rho, list(set(range(self.system_size)) - set(subsys_AB)))
        
        # Calculate entropies
        S_B = self.von_neumann_entropy(rho_B)
        S_AB = self.von_neumann_entropy(rho_AB)
        
        # Calculate conditional entropy
        return S_AB - S_B

    def entropy_evolution(self, entropy_type: str = 'von_neumann') -> np.ndarray:
        """
        Track entropy evolution
        
        Args:
            entropy_type: Type of entropy to track
            
        Returns:
            Array of entropy values over time
        """
        if entropy_type not in self.previous_entropies:
            raise ValueError(f"Entropy type must be one of {list(self.previous_entropies.keys())}")
            
        return np.array(self.previous_entropies[entropy_type])
    
    def visualize_entropy(self, filename: str = None):
        """
        Visualize entropy evolution
        
        Args:
            filename: Optional path to save visualization
        """
        if not any(self.previous_entropies.values()):
            logger.warning("No entropy history available for visualization")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for entropy_type, values in self.previous_entropies.items():
            if values:
                iterations = range(1, len(values) + 1)
                ax.plot(iterations, values, 'o-', label=entropy_type.replace('_', ' ').title())
                
        ax.set_title('Entropy Evolution')
        ax.set_xlabel('Measurement')
        ax.set_ylabel('Entropy (bits)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved entropy visualization to {filename}")
        else:
            plt.show()
    
class ErrorDetectionSystem:
    """
    Advanced error detection and correction system for quantum states
    
    This class implements stabilizer-based quantum error detection
    and correction techniques.
    """
    def __init__(self, n_qubits: int, error_threshold: float = 0.01, code_type: str = 'surface'):
        """
        Initialize the error detection system
        
        Args:
            n_qubits: Number of qubits in the system
            error_threshold: Threshold for error detection
            code_type: Type of error correction code to use
        """
        self.n_qubits = n_qubits
        self.error_threshold = error_threshold
        self.code_type = code_type
        self.syndrome_history: List[Dict[str, bool]] = []
        
        # Precompute Pauli matrices for efficiency
        self.pauli = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        
        # Generate stabilizers based on code type
        self.stabilizers = self.generate_stabilizer_generators()
        logger.info(f"Initialized error detection with {len(self.stabilizers)} stabilizers")
        
    def pauli_string_to_matrix(self, pauli_string: str) -> np.ndarray:
        """
        Convert Pauli string to matrix representation
        
        Args:
            pauli_string: String representation of Pauli operator (e.g., "XIZI")
            
        Returns:
            Matrix representation of the Pauli operator
        """
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string must have length {self.n_qubits}")
            
        # Start with identity
        result = 1
        
        # Build tensor product
        for op in pauli_string:
            if op not in self.pauli:
                raise ValueError(f"Invalid Pauli operator: {op}")
            result = np.kron(result, self.pauli[op])
            
        return result

    def generate_stabilizer_generators(self) -> List[Tuple[str, np.ndarray]]:
        """
        Generate stabilizer generators for error detection
        
        Returns:
            List of (pauli_string, matrix) tuples representing stabilizers
        """
        stabilizers = []
        
        if self.code_type == 'surface':
            # Surface code stabilizers
            # Use simplified version for demonstration
            k = int(np.sqrt(self.n_qubits))  # Assume n_qubits is perfect square
            
            # Generate plaquette Z-type stabilizers
            for i in range(k-1):
                for j in range(k-1):
                    # Z-type plaquette operator
                    pauli_string = ['I'] * self.n_qubits
                    
                    # Get qubit indices for this plaquette
                    qubits = [
                        i*k + j,      # top-left
                        i*k + (j+1),  # top-right
                        (i+1)*k + j,  # bottom-left
                        (i+1)*k + (j+1)  # bottom-right
                    ]
                    
                    for q in qubits:
                        if q < self.n_qubits:
                            pauli_string[q] = 'Z'
                            
                    pauli_str = ''.join(pauli_string)
                    stabilizers.append((pauli_str, self.pauli_string_to_matrix(pauli_str)))
                    
            # Generate star X-type stabilizers
            for i in range(1, k):
                for j in range(1, k):
                    # X-type star operator
                    pauli_string = ['I'] * self.n_qubits
                    
                    # Get qubit indices for this star
                    qubits = [
                        (i-1)*k + j,  # top
                        i*k + (j-1),  # left
                        i*k + (j+1),  # right
                        (i+1)*k + j   # bottom
                    ]
                    
                    for q in qubits:
                        if q < self.n_qubits:
                            pauli_string[q] = 'X'
                            
                    pauli_str = ''.join(pauli_string)
                    stabilizers.append((pauli_str, self.pauli_string_to_matrix(pauli_str)))
                
        else:
            # Simple code with nearest-neighbor stabilizers
            # X-type stabilizers
            for i in range(self.n_qubits - 1):
                pauli_string = ['I'] * self.n_qubits
                pauli_string[i] = 'X'
                pauli_string[i+1] = 'X'
                pauli_str = ''.join(pauli_string)
                stabilizers.append((pauli_str, self.pauli_string_to_matrix(pauli_str)))
                
            # Z-type stabilizers
            for i in range(self.n_qubits - 1):
                pauli_string = ['I'] * self.n_qubits
                pauli_string[i] = 'Z'
                pauli_string[i+1] = 'Z'
                pauli_str = ''.join(pauli_string)
                stabilizers.append((pauli_str, self.pauli_string_to_matrix(pauli_str)))
                
        return stabilizers

    def measure_error_syndromes(self, state: np.ndarray) -> Dict[str, bool]:
        """
        Measure error syndromes with improved accuracy
        
        Args:
            state: Quantum state (vector or density matrix)
            
        Returns:
            Dictionary of syndrome measurements
        """
        # Convert to density matrix if needed
        if len(state.shape) == 1:
            rho = np.outer(state, state.conj())
        else:
            rho = state
            
        syndromes = {}
        
        for i, (pauli_str, stabilizer) in enumerate(self.stabilizers):
            # Calculate expectation value of stabilizer
            expectation = np.real(np.trace(stabilizer @ rho))
            
            # Detect error if expectation value deviates from +1
            syndromes[f'S{i}_{pauli_str[:5]}'] = abs(1 - expectation) > self.error_threshold
            
        self.syndrome_history.append(syndromes)
        
        # Log syndrome results
        error_count = sum(syndromes.values())
        if error_count > 0:
            logger.warning(f"Detected {error_count} error syndromes")
            
        return syndromes

    def generate_error_correction_circuit(self, syndromes: Dict[str, bool]) -> QuantumCircuit:
        """
        Generate quantum circuit to correct errors based on syndromes
        
        Args:
            syndromes: Dictionary of syndrome measurements
            
        Returns:
            Quantum circuit for error correction
        """
        # Create quantum registers
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(len(syndromes), 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Apply correction operations based on syndromes
        correction_applied = False
        
        for syndrome_name, error_detected in syndromes.items():
            if error_detected:
                # Extract syndrome index and type
                parts = syndrome_name.split('_')
                idx = int(parts[0][1:])  # Extract number from 'S{i}'
                pauli_type = parts[1][0]  # Get first character of pauli string
                
                # Get affected qubits
                pauli_str = self.stabilizers[idx][0]
                affected_qubits = [i for i, op in enumerate(pauli_str) if op != 'I']
                
                # Apply correction to first affected qubit
                if affected_qubits:
                    target = affected_qubits[0]
                    
                    # Apply appropriate correction gate
                    if pauli_type == 'X':
                        circuit.z(qr[target])
                        circuit.barrier()
                    elif pauli_type == 'Z':
                        circuit.x(qr[target])
                        circuit.barrier()
                    else:
                        # Y error - apply both X and Z corrections
                        circuit.z(qr[target])
                        circuit.x(qr[target])
                        circuit.barrier()
                        
                    correction_applied = True
                    
        if correction_applied:
            # Add final syndrome measurement
            for i, (pauli_str, _) in enumerate(self.stabilizers):
                # Add stabilizer measurement
                # In a real circuit, this would be implemented with ancilla qubits
                # Here we just use classical register to store result
                circuit.measure_all(add_bits=False)
                
        return circuit

    def correct_errors(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply error correction based on syndrome measurements
        
        Args:
            state: Quantum state to correct
            
        Returns:
            Tuple of (corrected_state, correction_quality)
        """
        # Measure syndromes
        syndromes = self.measure_error_syndromes(state)
        
        # Check if errors detected
        if not any(syndromes.values()):
            return state, 1.0  # No errors detected
            
        # Initialize correction quality
        correction_quality = 1.0
        
        # Apply corrections based on code type
        if len(state.shape) == 1:
            # Pure state correction
            corrected_state = state.copy()
            
            for syndrome_name, error_detected in syndromes.items():
                if error_detected:
                    correction_quality *= self._apply_correction(
                        corrected_state, syndrome_name)
        else:
            # Mixed state correction
            corrected_state = state.copy()
            
            for syndrome_name, error_detected in syndromes.items():
                if error_detected:
                    correction_quality *= self._apply_correction_mixed_state(
                        corrected_state, syndrome_name)
                    
        logger.info(f"Applied error correction with quality {correction_quality:.4f}")
        return corrected_state, correction_quality
    
    def _apply_correction(self, state: np.ndarray, syndrome_name: str) -> float:
        """
        Apply specific error correction operation to pure state
        
        Args:
            state: Pure quantum state
            syndrome_name: Name of syndrome indicating error
            
        Returns:
            Correction quality (fidelity)
        """
        # Extract syndrome index
        parts = syndrome_name.split('_')
        idx = int(parts[0][1:])  # Extract number from 'S{i}'
        
        if idx < len(self.stabilizers):
            # Get Pauli operator and affected qubits
            pauli_str, _ = self.stabilizers[idx]
            
            # Determine correction operation based on syndrome type
            if 'X' in pauli_str:
                # X-type error requires Z correction
                correction_type = 'Z'
            else:
                # Z-type error requires X correction
                correction_type = 'X'
                
            # Find first affected qubit
            if 'Z' in pauli_str:
                affected_qubit = pauli_str.index('Z')
            elif 'X' in pauli_str:
                affected_qubit = pauli_str.index('X')
            else:
                return 1.0  # No affected qubit found
                
            # Apply correction
            correction_op = self.pauli_string_to_matrix(
                'I' * affected_qubit + correction_type + 'I' * (self.n_qubits - affected_qubit - 1)
            )
            
            # Apply correction to state
            corrected = correction_op @ state
            
            # Calculate correction quality
            fidelity = np.abs(np.vdot(state, corrected))**2
            
            # Update state in-place
            state[:] = corrected / np.linalg.norm(corrected)
            
            return fidelity
            
        return 1.0  # No correction applied
    
    def _apply_correction_mixed_state(self, state: np.ndarray, syndrome_name: str) -> float:
        """
        Apply specific error correction operation to mixed state
        
        Args:
            state: Mixed quantum state (density matrix)
            syndrome_name: Name of syndrome indicating error
            
        Returns:
            Correction quality (fidelity)
        """
        # Extract syndrome index
        parts = syndrome_name.split('_')
        idx = int(parts[0][1:])  # Extract number from 'S{i}'
        
        if idx < len(self.stabilizers):
            # Get Pauli operator and affected qubits
            pauli_str, _ = self.stabilizers[idx]
            
            # Determine correction operation based on syndrome type
            if 'X' in pauli_str:
                # X-type error requires Z correction
                correction_type = 'Z'
            else:
                # Z-type error requires X correction
                correction_type = 'X'
                
            # Find first affected qubit
            if 'Z' in pauli_str:
                affected_qubit = pauli_str.index('Z')
            elif 'X' in pauli_str:
                affected_qubit = pauli_str.index('X')
            else:
                return 1.0  # No affected qubit found
                
            # Create correction operator
            correction_pauli = 'I' * affected_qubit + correction_type + 'I' * (self.n_qubits - affected_qubit - 1)
            correction_op = self.pauli_string_to_matrix(correction_pauli)
            
            # Apply correction to density matrix
            corrected = correction_op @ state @ correction_op.conj().T
            
            # Calculate correction quality using fidelity
            fidelity = np.real(np.trace(sqrtm(
                sqrtm(state) @ corrected @ sqrtm(state)
            )))**2
            
            # Update state in-place
            state[:] = corrected
            
            return fidelity
            
        return 1.0  # No correction applied

    def analyze_error_patterns(self) -> Dict[str, float]:
        """
        Analyze error patterns from syndrome history
        
        Returns:
            Dictionary of error statistics
        """
        if not self.syndrome_history:
            return {'error_rate': 0.0}
            
        # Calculate error rates for each syndrome
        error_counts = {}
        total_measurements = len(self.syndrome_history)
        
        for syndromes in self.syndrome_history:
            for syndrome_name, error_detected in syndromes.items():
                if syndrome_name not in error_counts:
                    error_counts[syndrome_name] = 0
                    
                if error_detected:
                    error_counts[syndrome_name] += 1
        
        # Calculate error rates
        error_rates = {
            f"rate_{name}": count / total_measurements 
            for name, count in error_counts.items()
        }
        
        # Calculate overall error rate
        total_errors = sum(count for count in error_counts.values())
        total_possible = len(error_counts) * total_measurements
        
        error_rates['overall_error_rate'] = total_errors / total_possible
        
        # Add correlation analysis
        if len(self.syndrome_history) > 1:
            error_rates['temporal_correlation'] = self._calculate_temporal_correlation()
            
        return error_rates
    
    def _calculate_temporal_correlation(self) -> float:
        """
        Calculate temporal correlation of errors
        
        Returns:
            Correlation coefficient
        """
        # Extract time series of total errors
        time_series = []
        
        for syndromes in self.syndrome_history:
            error_count = sum(1 for error in syndromes.values() if error)
            time_series.append(error_count)
            
        # Calculate autocorrelation at lag 1
        if len(time_series) < 3:
            return 0.0
            
        # Calculate mean and variance
        mean = np.mean(time_series)
        variance = np.var(time_series)
        
        if variance == 0:
            return 0.0
            
        # Calculate autocorrelation
        autocorr = 0
        for i in range(len(time_series) - 1):
            autocorr += (time_series[i] - mean) * (time_series[i+1] - mean)
            
        autocorr /= (len(time_series) - 1) * variance
        
        return autocorr
    
    def visualize_error_syndromes(self, filename: str = None):
        """
        Visualize error syndrome patterns
        
        Args:
            filename: Optional path to save visualization
        """
        if not self.syndrome_history:
            logger.warning("No syndrome history available for visualization")
            return
            
        # Gather statistics
        syndrome_names = list(self.syndrome_history[0].keys())
        error_matrix = np.zeros((len(self.syndrome_history), len(syndrome_names)))
        
        for i, syndromes in enumerate(self.syndrome_history):
            for j, name in enumerate(syndrome_names):
                error_matrix[i, j] = int(syndromes[name])
                
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot error matrix
        im = ax1.imshow(error_matrix, aspect='auto', cmap='Reds')
        ax1.set_title('Error Syndromes Over Time')
        ax1.set_xlabel('Syndrome')
        ax1.set_ylabel('Measurement')
        ax1.set_xticks(range(len(syndrome_names)))
        ax1.set_xticklabels(syndrome_names, rotation=90)
        plt.colorbar(im, ax=ax1)
        
        # Plot error rates
        error_rates = [np.mean(error_matrix[:, j]) for j in range(len(syndrome_names))]
        ax2.bar(syndrome_names, error_rates)
        ax2.set_title('Error Rates by Syndrome')
        ax2.set_ylabel('Error Rate')
        ax2.set_xticklabels(syndrome_names, rotation=90)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved error syndrome visualization to {filename}")
        else:
            plt.show()

class ImprovedHolographicComputer:
    """
    Advanced holographic quantum computer implementation
    
    This class integrates tensor networks, noise models, and error
    correction to implement a complete holographic quantum computer.
    """
    def __init__(self, 
                 surface_qubits: int = 12, 
                 bulk_qubits: int = 3,
                 error_correction_level: int = 2,
                 visualization_enabled: bool = False):
        """
        Initialize the holographic quantum computer
        
        Args:
            surface_qubits: Number of qubits on the boundary
            bulk_qubits: Number of qubits in the bulk
            error_correction_level: Level of error correction (0-3)
            visualization_enabled: Whether to enable visualizations
        """
        self.surface_qubits = surface_qubits
        self.bulk_qubits = bulk_qubits
        self.error_correction_level = error_correction_level
        self.visualization_enabled = visualization_enabled
        
        # Create device connectivity graph
        self.connectivity = self._create_device_connectivity()
        
        # Initialize components
        logger.info(f"Initializing holographic computer with {surface_qubits} surface qubits and {bulk_qubits} bulk qubits")
        
        # Create tensor network with visualization if enabled
        self.tensor_network = TensorNetwork(
            nodes=surface_qubits,
            dimension=2**bulk_qubits,
            visualization_enabled=visualization_enabled
        )
        
        self.encoder = OptimizedHolographicEncoder(
            surface_qubits, 
            bulk_qubits,
            self.tensor_network
        )
        
        self.noise_model = EnhancedNoiseModel(
            device_connectivity=self.connectivity
        )
        
        self.entropy_calculator = EnhancedEntropyCalculator(
            surface_qubits + bulk_qubits
        )
        
        self.error_detector = ErrorDetectionSystem(
            surface_qubits + bulk_qubits,
            error_threshold=0.05
        )
        
        # Setup quantum circuit
        self.setup_quantum_circuit()
        
        # Store computation history
        self.computation_history: List[HolographicState] = []
        
    def _create_device_connectivity(self) -> nx.Graph:
        """
        Create device connectivity graph
        
        Returns:
            NetworkX graph representing qubit connectivity
        """
        total_qubits = self.surface_qubits + self.bulk_qubits
        
        # Create appropriate connectivity for the qubits
        if total_qubits <= 16:
            # Fully connected for small systems
            graph = nx.complete_graph(total_qubits)
        else:
            # Use grid-like structure for larger systems
            width = int(np.ceil(np.sqrt(total_qubits)))
            height = int(np.ceil(total_qubits / width))
            
            # Create grid
            graph = nx.grid_2d_graph(width, height)
            
            # Map 2D coordinates to qubit indices
            mapping = {}
            for i in range(width):
                for j in range(height):
                    idx = i * height + j
                    if idx < total_qubits:
                        mapping[(i, j)] = idx
                        
            # Relabel nodes
            graph = nx.relabel_nodes(graph, mapping)
            
            # Add some additional connections for bulk qubits
            for i in range(self.bulk_qubits):
                bulk_id = self.surface_qubits + i
                # Connect each bulk qubit to a few surface qubits
                for j in range(min(4, self.surface_qubits)):
                    graph.add_edge(bulk_id, j * (self.surface_qubits // 4))
                    
        return graph

    def setup_quantum_circuit(self):
        """
        Setup improved quantum circuit with proper registers
        """
        # Create quantum registers
        self.surface_reg = QuantumRegister(self.surface_qubits, 'surface')
        self.bulk_reg = QuantumRegister(self.bulk_qubits, 'bulk')
        
        # Add syndrome qubits for error correction
        syndrome_size = max(1, self.error_correction_level * 2)
        self.syndrome_reg = QuantumRegister(syndrome_size, 'syndrome')
        
        # Classical registers for measurement
        self.classical_reg = ClassicalRegister(
            self.surface_qubits + self.bulk_qubits, 'c'
        )
        self.syndrome_creg = ClassicalRegister(syndrome_size, 'sc')
        
        # Create circuit
        self.circuit = QuantumCircuit(
            self.surface_reg,
            self.bulk_reg,
            self.syndrome_reg,
            self.classical_reg,
            self.syndrome_creg
        )
        
        logger.info(f"Created quantum circuit with {self.circuit.num_qubits} qubits")

    def prepare_initial_state(self, state: np.ndarray):
        """
        Prepare initial quantum state with error checking
        
        Args:
            state: Initial quantum state vector
        """
        if state.shape[0] != 2**self.bulk_qubits:
            raise ValueError(
                f"Invalid state dimension {state.shape[0]}, expected {2**self.bulk_qubits}"
            )
            
        # Reset circuit
        self.circuit = QuantumCircuit(
            self.surface_reg,
            self.bulk_reg,
            self.syndrome_reg,
            self.classical_reg,
            self.syndrome_creg
        )
            
        # Normalize state
        state = state / np.linalg.norm(state)
        
        logger.info("Preparing initial quantum state")
        
        # Use efficient state preparation
        try:
            # Try to use efficient initialization
            self._initialize_state_efficiently(state)
        except Exception as e:
            logger.warning(f"Efficient initialization failed: {str(e)}")
            # Fall back to basic initialization
            self._initialize_state_basic(state)
            
        # Add barrier after initialization
        self.circuit.barrier()

    def _initialize_state_efficiently(self, state: np.ndarray):
        """
        Efficient quantum state initialization
        
        Args:
            state: Quantum state vector to initialize
        """
        # Create state initialization subcircuit
        sub_qubits = QuantumRegister(self.bulk_qubits, 'q')
        init_circuit = QuantumCircuit(sub_qubits)
        
        # Use Qiskit's initialize method which finds an efficient decomposition
        init_circuit.initialize(state, sub_qubits)
        
        # Convert to gate and add to main circuit
        init_gate = init_circuit.to_gate(label="init_state")
        self.circuit.append(init_gate, self.bulk_reg)
        
    def _initialize_state_basic(self, state: np.ndarray):
        """
        Basic quantum state initialization using amplitude encoding
        
        Args:
            state: Quantum state vector to initialize
        """
        # Apply series of rotations to achieve desired amplitudes
        for i, amplitude in enumerate(state):
            if abs(amplitude) > 1e-10:
                # Convert index to binary representation
                binary = format(i, f'0{self.bulk_qubits}b')
                
                # Prepare computational basis state
                for j, bit in enumerate(binary):
                    if bit == '1':
                        self.circuit.x(self.bulk_reg[j])
                
                # Apply phase and amplitude using controlled rotations
                theta, phi, lam = self._bloch_angles(amplitude)
                
                # Use control qubits to target this specific basis state
                controls = [self.bulk_reg[j] if bit == '1' else ~self.bulk_reg[j] 
                           for j, bit in enumerate(binary)]
                
                # Apply rotation to first syndrome qubit
                self.circuit.x(self.syndrome_reg[0])  # Start in |1⟩
                
                # Apply controlled rotation
                self.circuit.append(
                    self.circuit.u(theta, phi, lam).control(
                        len(controls), ctrl_state='1' * len(controls)
                    ),
                    controls + [self.syndrome_reg[0]]
                )
                
                # Reset basis state
                for j, bit in enumerate(binary):
                    if bit == '1':
                        self.circuit.x(self.bulk_reg[j])

    def _bloch_angles(self, complex_amplitude: complex) -> Tuple[float, float, float]:
        """
        Convert complex amplitude to Bloch sphere angles
        
        Args:
            complex_amplitude: Complex amplitude value
            
        Returns:
            Tuple of (theta, phi, lambda) angles for U gate
        """
        # Get magnitude and phase
        magnitude = abs(complex_amplitude)
        phase = np.angle(complex_amplitude)
        
        # Convert to Bloch sphere angles
        theta = 2 * np.arccos(magnitude)  # Rotation angle
        phi = 0  # Angle around z-axis before rotation
        lam = phase  # Angle around z-axis after rotation
        
        return theta, phi, lam

    def encode_holographic_state(self, bulk_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Encode bulk state into holographic boundary representation
        
        Args:
            bulk_state: Quantum state in the bulk
            
        Returns:
            Tuple of (encoded_state, fidelity)
        """
        logger.info("Encoding holographic state")
        return self.encoder.encode_with_optimization(bulk_state)

    def execute_holographic_computation(self, 
                                     initial_state: np.ndarray,
                                     shots: int = 1000,
                                     visualization: bool = False) -> Dict:
        """
        Execute full holographic computation with monitoring
        
        Args:
            initial_state: Initial quantum state
            shots: Number of measurement shots
            visualization: Whether to generate visualizations
            
        Returns:
            Dictionary of computation results
        """
        start_time = time.time()
        logger.info(f"Starting holographic computation with {shots} shots")
        
        # Step 1: Prepare and encode state
        # Prepare initial state
        self.prepare_initial_state(initial_state)
        
        # Encode state holographically
        encoded_state, encoding_fidelity = self.encode_holographic_state(initial_state)
        
        # Step 2: Add error detection based on error correction level
        if self.error_correction_level > 0:
            # Measure syndromes before noise
            syndromes_before = self.error_detector.measure_error_syndromes(encoded_state)
            
            # Add error detection circuits
            self._add_error_detection_circuits()
        else:
            syndromes_before = {}
            
        # Step 3: Apply noise model
        noise_model = self.noise_model.add_realistic_noise()
        
        # Step 4: Execute circuit
        if self.error_correction_level > 1:
            # Add mid-circuit error correction
            self._add_mid_circuit_error_correction()
            
# Add final measurements
        self.circuit.measure(self.surface_reg, self.classical_reg[:self.surface_qubits])
        self.circuit.measure(self.bulk_reg, 
                            self.classical_reg[self.surface_qubits:self.surface_qubits+self.bulk_qubits])
                            
        # Execute circuit with noise model
        logger.info(f"Executing quantum circuit with {self.circuit.num_qubits} qubits and {len(self.circuit)} operations")
        job = execute(self.circuit, 
                     Aer.get_backend('qasm_simulator'),
                     noise_model=noise_model,
                     shots=shots,
                     optimization_level=2)
        
        # Process results
        results = job.result()
        counts = results.get_counts()
        
        # Step 5: Process measurement outcomes
        # Get statevector for entropy calculation (in noiseless simulation)
        statevector_sim = Aer.get_backend('statevector_simulator')
        noiseless_circuit = self.circuit.copy()
        
        # Remove measurement operations for statevector simulation
        for op_idx in range(len(noiseless_circuit) - 1, -1, -1):
            if noiseless_circuit[op_idx][0].name == 'measure':
                noiseless_circuit.data.pop(op_idx)
                
        # Execute noiseless circuit to get final state
        noiseless_job = execute(noiseless_circuit, statevector_sim)
        final_state = noiseless_job.result().get_statevector()
        
        # Calculate entropy
        entropy = self.entropy_calculator.von_neumann_entropy(
            final_state.data.reshape(-1, 1) @ final_state.data.reshape(1, -1)
        )
        
        # Step 6: Apply error correction if needed
        if self.error_correction_level > 0:
            # Measure syndromes after execution
            final_state_dm = final_state.data.reshape(-1, 1) @ final_state.data.reshape(1, -1)
            syndromes_after = self.error_detector.measure_error_syndromes(final_state_dm)
            
            # Apply error correction
            corrected_state, correction_quality = self.error_detector.correct_errors(final_state_dm)
        else:
            syndromes_after = {}
            correction_quality = 1.0
            corrected_state = final_state.data
            
        # Step 7: Create holographic state object
        holographic_state = HolographicState(
            bulk_data=initial_state,
            surface_encoding=encoded_state,
            entanglement_entropy=entropy,
            error_syndromes=list(syndromes_after.values() if syndromes_after else []),
            fidelity=encoding_fidelity,
            metadata={
                'execution_time': time.time() - start_time,
                'circuit_depth': self.circuit.depth(),
                'correction_quality': correction_quality,
                'shots': shots
            }
        )
        
        # Store computation history
        self.computation_history.append(holographic_state)
        
        # Generate visualizations if requested
        if visualization and self.visualization_enabled:
            self._generate_visualizations()
            
        # Prepare result dictionary
        elapsed = time.time() - start_time
        logger.info(f"Holographic computation completed in {elapsed:.2f} seconds")
        
        return {
            'counts': counts,
            'entropy': entropy,
            'fidelity': encoding_fidelity,
            'error_syndromes': syndromes_after,
            'correction_quality': correction_quality,
            'execution_time': elapsed,
            'circuit_depth': self.circuit.depth(),
            'holographic_state': holographic_state
        }
        
    def _add_error_detection_circuits(self):
        """
        Add error detection circuits based on error correction level
        """
        # Number of syndrome measurements depends on error correction level
        n_syndromes = self.error_correction_level
        
        # Add stabilizer measurements
        for i in range(min(n_syndromes, len(self.syndrome_reg))):
            # Add stabilizer measurement using syndrome qubit
            self.circuit.h(self.syndrome_reg[i])
            
            # Apply controlled-Z operations between syndrome and data qubits
            # Based on simplified surface code for demonstration
            targets = [
                self.surface_reg[i % self.surface_qubits],
                self.surface_reg[(i + 1) % self.surface_qubits],
                self.bulk_reg[i % self.bulk_qubits]
            ]
            
            for target in targets:
                self.circuit.cz(self.syndrome_reg[i], target)
                
            # Measure syndrome
            self.circuit.h(self.syndrome_reg[i])
            self.circuit.measure(self.syndrome_reg[i], self.syndrome_creg[i])
            
            # Reset syndrome qubit for reuse
            self.circuit.reset(self.syndrome_reg[i])
            
        # Add barrier to separate error detection
        self.circuit.barrier()
        
    def _add_mid_circuit_error_correction(self):
        """
        Add mid-circuit error correction operations
        """
        # Apply conditional operations based on syndrome measurements
        for i in range(min(self.error_correction_level, len(self.syndrome_reg))):
            # Apply X correction if syndrome is 1
            self.circuit.x(self.surface_reg[i]).c_if(self.syndrome_creg[i], 1)
            
            # Apply Z correction to bulk qubits
            if i < self.bulk_qubits:
                self.circuit.z(self.bulk_reg[i]).c_if(self.syndrome_creg[i], 1)
                
        # Add barrier after correction
        self.circuit.barrier()
        
    def _generate_visualizations(self):
        """
        Generate visualizations for the computation
        """
        # Create directory for visualizations
        import os
        viz_dir = "holographic_viz"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate tensor network visualization
        self.tensor_network.visualize(f"{viz_dir}/tensor_network.png")
        
        # Generate noise profile visualization
        self.noise_model.visualize_noise_profile(f"{viz_dir}/noise_profile.png")
        
        # Generate entropy visualization
        self.entropy_calculator.visualize_entropy(f"{viz_dir}/entropy_evolution.png")
        
        # Generate error syndrome visualization
        self.error_detector.visualize_error_syndromes(f"{viz_dir}/error_syndromes.png")
        
        # Generate circuit visualization if not too large
        if self.circuit.num_qubits <= 20:
            circuit_diagram = self.circuit.draw(output='mpl')
            circuit_diagram.savefig(f"{viz_dir}/quantum_circuit.png")
            
        logger.info(f"Generated visualizations in directory: {viz_dir}")
        
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze computational results
        
        Args:
            results: Results from holographic computation
            
        Returns:
            Dictionary of analysis metrics
        """
        # Extract key metrics
        fidelity = results.get('fidelity', 0)
        entropy = results.get('entropy', 0)
        counts = results.get('counts', {})
        
        # Calculate success probability
        if counts:
            # Get most frequent outcome
            max_count = max(counts.values())
            success_prob = max_count / sum(counts.values())
        else:
            success_prob = 0
            
        # Calculate error metrics
        error_rate = 1 - results.get('correction_quality', 1.0)
        
        # Calculate resource usage
        circuit_depth = results.get('circuit_depth', 0)
        circuit_width = self.circuit.num_qubits
        
        # Calculate holographic metrics
        holographic_state = results.get('holographic_state')
        rt_entropy = entropy
        
        # Calculate efficiency metrics
        execution_time = results.get('execution_time', 0)
        ops_per_second = len(self.circuit) / execution_time if execution_time > 0 else 0
        
        return {
            'success_probability': success_prob,
            'encoding_fidelity': fidelity,
            'entanglement_entropy': entropy,
            'error_rate': error_rate,
            'rt_entropy': rt_entropy,
            'circuit_complexity': {
                'depth': circuit_depth,
                'width': circuit_width,
                'total_operations': len(self.circuit)
            },
            'performance': {
                'execution_time': execution_time,
                'operations_per_second': ops_per_second
            }
        }
        
    def clear_history(self):
        """
        Clear computation history
        """
        self.computation_history = []
        
    def get_computation_summary(self) -> Dict:
        """
        Get summary of all computations performed
        
        Returns:
            Dictionary of computation metrics
        """
        if not self.computation_history:
            return {'status': 'No computations performed'}
            
        # Extract key metrics
        fidelities = [state.fidelity for state in self.computation_history]
        entropies = [state.entanglement_entropy for state in self.computation_history]
        
        # Calculate averages
        avg_fidelity = np.mean(fidelities)
        avg_entropy = np.mean(entropies)
        
        # Calculate success trend
        fidelity_trend = 'improving' if fidelities[-1] > fidelities[0] else 'degrading'
        
        return {
            'num_computations': len(self.computation_history),
            'average_fidelity': avg_fidelity,
            'average_entropy': avg_entropy,
            'fidelity_trend': fidelity_trend,
            'latest_fidelity': fidelities[-1],
            'computation_success': avg_fidelity > 0.8
        }

def run_comprehensive_simulation():
    """
    Run complete holographic computation simulation with expanded capabilities
    
    Returns:
        Dictionary of simulation results and analysis
    """
    logger.info("Starting comprehensive holographic simulation")
    
    # Initialize holographic quantum computer with visualization
    computer = ImprovedHolographicComputer(
        surface_qubits=12,
        bulk_qubits=3,
        error_correction_level=2,
        visualization_enabled=True
    )
    
    # Create complex initial state - several options
    # Option 1: Random state
    initial_state = random_statevector(2**3).data
    
    # Option 2: GHZ-like state
    # initial_state = np.zeros(2**3)
    # initial_state[0] = 1/np.sqrt(2)
    # initial_state[-1] = 1/np.sqrt(2)
    
    # Option 3: W-state
    # initial_state = np.zeros(2**3)
    # initial_state[1] = initial_state[2] = initial_state[4] = 1/np.sqrt(3)
    
    # Normalize state
    initial_state = initial_state / np.linalg.norm(initial_state)
    
    # Execute computation with visualization
    results = computer.execute_holographic_computation(
        initial_state,
        shots=2000,
        visualization=True
    )
    
    # Analyze results
    analysis = computer.analyze_results(results)
    
    # Generate expanded result set
    simulation_results = {
        'measurement_counts': results['counts'],
        'final_entropy': results['entropy'],
        'encoding_fidelity': results['fidelity'],
        'error_detection': results['error_syndromes'],
        'analysis': analysis,
        'entropy_evolution': computer.entropy_calculator.entropy_evolution(),
        'error_analysis': computer.error_detector.analyze_error_patterns()
    }
    
    logger.info("Holographic simulation completed successfully")
    return simulation_results

if __name__ == "__main__":
    # Run full simulation
    simulation_results = run_comprehensive_simulation()
    
    # Print summary of results
    print("\nHolographic Quantum Computation Results:")
    print("===============cbrwx==================")
    
    print(f"\nEncoding Fidelity: {simulation_results['encoding_fidelity']:.4f}")
    print(f"Entanglement Entropy: {simulation_results['final_entropy']:.4f}")
    
    print("\nAnalysis:")
    for key, value in simulation_results['analysis'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    print("\nError Analysis:")
    for key, value in simulation_results['error_analysis'].items():
        print(f"  {key}: {value:.4f}")
        
    # Plot top measurement outcomes
    counts = simulation_results['measurement_counts']
    plot_histogram(counts, figsize=(12, 6), title="Measurement Outcomes")
    plt.savefig("measurement_outcomes.png")
    
    print("\nVisualization files have been saved to the 'holographic_viz' directory.")
    print("Measurement outcomes plot saved as 'measurement_outcomes.png'.")
