
from typing import List, Tuple, Dict, Union, Literal, Any, Iterator
from collections import deque

from .term_base import (
    FittableFOMTerm, FOMTermType, FOMTermInstance, FOMTermClasses
)
from .term_gpr import SurrogateGPRTerm
from .term_gpc import InterestGPCTerm, InlierGPCTerm
from .term_mlp import MLPTerm
from .term_kde import OutlierKDETerm, FeatureKDETerm, TargetKDETerm
from .term_spatial import OutlierProximityTerm, SigmoidDensityTerm


class FOMTermAccessor:
    """
    This class enables accessing terms in FigureOfMerit.terms.term_name with
    autocompletion and supports dict-like iteration methods (items, values and
    __iter__).

    Note:
        The order of term names is carefully defined to enable a consitent
        fitting of terms in FOM class, respecting term dependencies.
    """

    surrogate_gpr: SurrogateGPRTerm
    interest_gpc: InterestGPCTerm
    inlier_gpc: InlierGPCTerm
    surrogate_mlp: MLPTerm
    outlier_kde: OutlierKDETerm
    feature_kde: FeatureKDETerm
    target_kde: TargetKDETerm
    outlier_proximity: OutlierProximityTerm
    sigmoid_density: SigmoidDensityTerm

    def __init__(self, terms: Dict[str, FOMTermInstance]):
        # Set terms as attributes
        for term_name, term_instance in terms.items():
            if term_name in self.__annotations__:
                setattr(self, term_name, term_instance)
            else:
                raise AttributeError(f"Unknown term: {term_name}")
    
        self.term_names = list(terms.keys())

        # Validate dependencies after initializing all terms
        self.validate_term_dependencies()

        # Sort term names in topological order to enable consistent fitting in FOM class
        self.term_names = self.topological_sort(self.term_names)

    def __getitem__(self, term_name: str) -> FOMTermInstance:
        """ Enables dict-like access to terms using square bracket notation. """
        if term_name not in self.term_names:
            raise KeyError(f"Term '{term_name}' not found")

        return getattr(self, term_name)

    def items(self) -> Iterator[Tuple[str, FOMTermInstance]]:
        """
        Returns an iterator of (term_name, term_instance) pairs.
        """
        return ((term_name, getattr(self, term_name)) for term_name in self.term_names)

    def values(self) -> Iterator[FOMTermInstance]:
        """
        Returns an iterator of all term instances.
        """
        return (getattr(self, term_name) for term_name in self.term_names)

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator of all term names.
        """
        return iter(self.term_names)

    @classmethod
    def is_valid_term_class(cls, TermClass: FOMTermType) -> bool:
        """
        Checks if a given input is a valid FOM term class.
        """
        return (
            isinstance(TermClass, type) and
            issubclass(TermClass, FOMTermClasses)
        )

    @classmethod
    def is_valid_term_name(cls, term_name: str) -> bool:
        """
        Checks if a given term name is defined and is a valid FOM term class.
        """
        if term_name not in cls.__annotations__:
            return False
        TermClass = cls.__annotations__[term_name]
        return cls.is_valid_term_class(TermClass)

    @classmethod
    def get_all_term_names(cls) -> List[str]:
        """ Returns a list of all possible term names. """
        return list(cls.__annotations__.keys())

    @classmethod
    def get_term_class(cls, term_name: str) -> FOMTermType:
        """ Returns the class for a given term name. """
        if not cls.is_valid_term_name(term_name):
            raise ValueError(f"'{term_name}' is not a valid FOM term class.")
        return cls.__annotations__[term_name]

    def validate_term_dependencies(self) -> None:
        """
        Validates that all dependencies of terms (if any) are present in self.term_names.
        
        Raises:
            ValueError: If a term has a dependency that is not in self.term_names.
        """
        for term_name, term in self.items():
            if isinstance(term, FittableFOMTerm):
                all_term_names = self.get_all_term_names()
                invalid_dependencies = set(term.dependencies) - set(all_term_names)
                if invalid_dependencies:
                    raise ValueError(
                        f"Term '{term_name}' has unavailable dependencies: {invalid_dependencies}. "
                        f"Available term names are: {all_term_names}"
                    )

    def topological_sort(self, term_names: List[str]) -> List[str]:
        """
        Perform a topological sort on given terms based on their dependencies.

        This function uses Kahn's algorithm for topological sorting. It employs a queue
        to process nodes (terms) in a specific order, ensuring that each term is processed
        only after all its dependencies have been handled.

        The queue serves several purposes:
        1. It initially stores all nodes with no dependencies (in-degree of 0).
        2. It maintains the order in which nodes should be processed, prioritizing
        nodes whose dependencies have all been satisfied.
        3. It enables breadth-first processing of the dependency graph.
        4. It helps in cycle detection: if the queue becomes empty before all nodes
        are processed, it indicates a cyclic dependency.

        The algorithm works as follows:
        1. Initialize the queue with all nodes that have no dependencies.
        2. While the queue is not empty:
            a. Dequeue a node and add it to the sorted output.
            b. For each node that depends on the current node:
                - Decrease its in-degree (number of unprocessed dependencies).
                - If its in-degree becomes 0, enqueue it.
        3. If all nodes are processed, return the sorted list.
        Otherwise, raise an error indicating a cyclic dependency.
        """

        # Create a graph representation
        graph = {term_name: set() for term_name in term_names}
        for term_name in term_names:
            term = getattr(self, term_name)
            if isinstance(term, FittableFOMTerm):
                # Add edges (dependencies) only for fittable terms
                # Consider only active dependencies
                active_dependencies = [dep for dep in term.dependencies if dep in term_names]
                graph[term_name].update(active_dependencies)

        # Calculate in_degree, the number of edges j != i pointing to node i
        # Also interpreted as number of terms that depend on term i
        in_degree = {node: len(graph[node]) for node in graph}

        # Initialize the queue with nodes that have no dependencies
        queue = deque([node for node in graph if in_degree[node] == 0])
        sorted_term_names = []

        # Perform topological sort
        while queue:
            # Process the next term in the queue
            node = queue.popleft()
            sorted_term_names.append(node)

            # Reduce the in-degree of this node's neighbors
            for dependent in graph:
                if node in graph[dependent]:
                    in_degree[dependent] -= 1
                    # If neighbor now has no dependencies, add it to the queue
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(sorted_term_names) != len(graph):
            print("Debug: Sorted terms:", sorted_term_names)
            print("Debug: Remaining in-degrees:", {k: v for k, v in in_degree.items() if v > 0})
            print("Debug: Graph:", graph)
            raise ValueError("Cyclic dependencies detected in terms")

        # Reverse the list to get the desired order
        return sorted_term_names
