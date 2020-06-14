import heapq


class NodeHeap:
    """Heap of nodes that allows update

    Inspired from https://docs.python.org/3.7/library/heapq.html#priority-queue-implementation-notes
    """
    REMOVED = -1

    def __init__(self):
        self.elements = []
        self.node_finder = {}

    def add_node(self, node, weight):
        """Adds or updates a node with priority sets to its weight"""
        if node == NodeHeap.REMOVED:
            raise Error(
                f'NodeHeap cannot deal with node value={NodeHeap.REMOVED}. This value is used as a default code.')

        if node in self.node_finder:
            self.remove_node(node)

        entry = [weight, node]
        self.node_finder[node] = entry
        heapq.heappush(self.elements, entry)

    def remove_node(self, node):
        entry = self.node_finder.pop(node)
        entry[-1] = NodeHeap.REMOVED

    def pop_node(self):
        while len(self.elements) > 0:
            priority, node = heapq.heappop(self.elements)
            if node != NodeHeap.REMOVED:
                del self.node_finder[node]
                return priority, node
        raise KeyError('pop from an empty priority queue')


def core_number_weighted(g):
    """Computes the hierarchy of k-cores

    From "Generalized Cores" V. Batagelj, M. Zaveršnik (2002)
    URL: https://arxiv.org/abs/cs/0202039
    Section: 3.2  Determining the hierarchy of p-cores
    """
    graph = g.copy()
    core_number = {}

    sorted_nodes = NodeHeap()
    [sorted_nodes.add_node(node, weight) for node, weight in graph.degree(weight='weight')]

    not_empty = True
    while not_empty:
        try:
            node_priority, current_node = sorted_nodes.pop_node()

            # Computation of the core number
            core_number[current_node] = node_priority

            # Recovering neighbors of the current node
            neighbors = graph.neighbors(current_node)
            graph.remove_node(current_node)

            for n in neighbors:
                new_priority_n = max(core_number[current_node], graph.degree(n, weight='weight'))
                sorted_nodes.add_node(n, new_priority_n)
        except KeyError as e:
            not_empty = False

    return core_number
