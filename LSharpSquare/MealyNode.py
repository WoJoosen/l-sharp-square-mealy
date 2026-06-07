class MealyNode:
    _id_counter = 0
    __slots__ = ['id', 'successors', 'parent', 'input_to_parent', 'access_sequence', 'leads_to_known']

    def __init__(self, parent=None):
        MealyNode._id_counter += 1
        self.id = MealyNode._id_counter
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None
        self.access_sequence = []
        self.leads_to_known = False

    def __hash__(self):
        return hash(self.id)

    def set_output_state(self, output):
        """Mark this node as having a known output from the transition leading to it"""
        if output is not None:
            self.leads_to_known = True
            node = self
            while node.parent is not None and not node.parent.leads_to_known:
                node = node.parent
                node.leads_to_known = True

    def add_successor(self, input_val, output_val, successor_node):
        """ Adds a successor node to the current node based on input and output """
        self.successors[input_val] = (output_val, successor_node)
        self.successors[input_val][1].parent = self
        self.successors[input_val][1].input_to_parent = input_val
        self.successors[input_val][1].set_output_state(output_val)
        self.successors[input_val][1].access_sequence = self.access_sequence + [input_val]

    def get_successor(self, input_val):
        """ Returns the successor node for the given input """
        if input_val in self.successors:
            return self.successors[input_val][1]
        return None

    def get_output(self, input_val):
        """ Returns the output for the given input """
        if input_val in self.successors:
            return self.successors[input_val][0]
        return None

    def extend_and_get(self, inp, output):
        """ Extend the node with a new successor and return the successor node """
        if inp in self.successors:
            out = self.successors[inp][0]
            if out != output:
                raise Exception(
                    f"observation not consistent with tree with output from tree: {out} and output from call: {output}")
            return self.successors[inp][1]
        successor_node = MealyNode(parent=self)
        self.add_successor(inp, output, successor_node)
        successor_node.input_to_parent = inp
        return successor_node

    @property
    def id_counter(self):
        return self._id_counter