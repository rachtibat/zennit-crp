import torch
from typing import List


class GraphNode:
    """
    Contains meta information about a node in a PyTorch jit graph
    """

    def __init__(self, node):
        self.id = node.__repr__()
        self.scopeName = node.scopeName()

        if len(self.scopeName) == 0:
            self.scopeName = node.kind()

        self.output_nodes = []
        self.input_nodes = []
        self.is_layer = False
        self.input_layers = []

        if "aten" in node.kind():
            self.layer_name = self.scopeName.split("__module.")[-1]
            self.is_layer = True


class ModelGraph:
    """
    This class contains meta information about layer connections inside the PyTorch model.

    Use `find_input_layers` method to get the input layers of a specific nn.Module
    """

    def __init__(self, input_nodes):

        self.id_node_map = {}
        self.layer_node_map = {}

        for node in input_nodes:
            self._add_node(node)

    def _add_node(self, node):
        """
        Add node to intern dictionaries
        """

        node_id = node.__repr__()
        if node_id not in self.id_node_map:
            node_obj = GraphNode(node)
            self.id_node_map[node_id] = node_obj

            if node_obj.is_layer:
                self.layer_node_map[node_obj.layer_name] = node_obj

        return self.id_node_map[node_id]

    def _add_connection(self, in_node, out_node):
        """
        Add an entry for the connection between input_node and end_node
        """
        layer_node_in = self._add_node(in_node)
        layer_node_out = self._add_node(out_node)

        new_connection = False

        if layer_node_in not in layer_node_out.input_nodes:
            layer_node_out.input_nodes.append(layer_node_in)
            new_connection = True
        if layer_node_out not in layer_node_in.output_nodes:
            layer_node_in.output_nodes.append(layer_node_out)
            new_connection = True

        return new_connection

    def set_layer_names(self, layer_names: list):

        self.layer_names = layer_names
        for name in layer_names:
            # cache results
            self.find_input_layers(name)

    def find_input_layers(self, layer_name: str) -> List:
        """
        Returns all layer names that are connected to the input of `layer_name`.
        The method returns only layers, that were supplied as `layer_names` argument
        to the `trace_model_graph` function. If you wish to change available layers,
        please modify the self.layer_names attribute.

        Parameters:
        ----------
        layer_name: str
            name of torch.nn.Module
        """

        if layer_name not in self.layer_node_map:
            raise KeyError(f"{layer_name} does not exist")

        root_node = self.layer_node_map[layer_name]

        if len(root_node.input_layers) == 0:
            # cache results
            layers = self._recursive_search(root_node)
            root_node.input_layers = layers
            return layers

        else:
            return root_node.input_layers

    def _recursive_search(self, node_obj):

        found_layers = []
        for g_node in node_obj.input_nodes:

            if g_node.is_layer and g_node.layer_name in self.layer_names:
                found_layers.append(g_node.layer_name)

            else:
                found_layers.extend(self._recursive_search(g_node))

        return found_layers

    def __str__(self):

        model_string = ""

        for layer in self.id_node_map.values():
            model_string += layer.scopeName + " -> "
            for next_l in layer.output_nodes:
                model_string += next_l.scopeName + ", "

            if len(layer.output_nodes) == 0:
                model_string += " end"
            else:
                model_string += "\n"

        return model_string


def trace_model_graph(model, sample: torch.Tensor, layer_names: List[str], debug=False) -> ModelGraph:
    """"
    As pytorch does not trace the model structure like tensorflow, we need to do it ourselves.
    Thus, this function generates a model graph - a summary - how all nn.Module are connected with each other.

    Parameters:
    ----------
        model: torch.nn.Module
        sample: torch.Tensor
            An examplary input. Used to trace the model with torch.jit.trace
        layer_names: list of strings
            List of all layer names that should be accessible in the model graph summary
        debug: boolean
            If True, returns the tarced inlined_graph of torch.jit

    Returns:
    -------
    ModelGraph: obj
        Object that contains meta information about the connection of modules.
        Use the `find_input_layers` method to get the input layers of a specific nn.Module.

    """

    # we use torch.jit to record the connections of all tensors
    traced = torch.jit.trace(model, (sample,), check_trace=False)
    # inlined_graph returns a suitable presentation of the traced model
    graph = traced.inlined_graph

    if debug is True:
        dump_pytorch_graph(graph)

    """
    We search for all input nodes where we could start a recursive travers through the network graph.
    First, we concatenate all input and output tensor ids for each node as they are spread out in the original
    torch.jit representation. Then, we search for a node with input tensors that have no connection to the output
    of another node. This node is an input node per definition.
    """
    node_inputs, node_outputs = _collect_node_inputs_and_outputs(graph)
    input_nodes = _get_input_nodes(graph, node_inputs, node_outputs)

    # initialize a model representation where we save the results
    MG = ModelGraph(input_nodes)

    # start recursive decoding of torch.jit graph
    for node in input_nodes:
        _build_graph_recursive(MG, graph, node)

    MG.set_layer_names(layer_names)

    # explicitly free gpu ram
    del traced, graph

    return MG


def _build_graph_recursive(MG: ModelGraph, graph, in_node):
    """
    Recursive function traverses the graph constructed by torch.jit.trace
    and records the graph structure inside our ModelGraph class
    """

    node_outputs = [i.unique() for i in in_node.outputs()]
    next_nodes = _find_next_nodes(graph, node_outputs)

    if len(next_nodes) == 0:
        return

    for node in next_nodes:
        new_connection = MG._add_connection(in_node, node)

        if new_connection:
            _build_graph_recursive(MG, graph, node)
        else:
            return


def _find_next_nodes(graph, node_outputs):
    """
    Helper function for build_graph_recursive.
    """

    next_nodes = []

    for node in graph.nodes():

        node_inputs = [i.unique() for i in node.inputs()]
        if set(node_inputs) & set(node_outputs):
            next_nodes.append(node)

    return next_nodes


def _collect_node_inputs_and_outputs(graph):
    """
    Helper function to get all tensor ids of the input and output of each node.
    Used to retrieve the input layers of the model.
    """

    layer_inputs = {}
    layer_outputs = {}

    for node in graph.nodes():
        # "aten" nodes are torch.nn.Modules
        if "aten" in node.kind():

            name = node.scopeName()

            if name not in layer_inputs:
                layer_inputs[name] = []
                layer_outputs[name] = []

            [layer_inputs[name].append(i.unique()) for i in node.inputs()]
            [layer_outputs[name].append(i.unique()) for i in node.outputs()]

    return layer_inputs, layer_outputs


def _get_input_nodes(graph, layer_inputs: dict, layer_outputs: dict):
    """
    Returns input nodes of jit graph.
    Used to retrieve the input layers of the model.
    """

    input_nodes = []

    for node in graph.nodes():
        # "aten" describes all real layers
        if "aten" in node.kind():

            name = node.scopeName()

            node_inputs = layer_inputs[name]
            # if its inputs are not ourputs of other modules -> an input node
            if not _find_overlap_with_output(node_inputs, layer_outputs):
                input_nodes.append(node)

    return input_nodes


def _find_overlap_with_output(node_inputs: list, layer_outputs: dict):
    """
    More efficient version of _find_next_nodes.
    """

    for name in layer_outputs:

        node_outputs = layer_outputs[name]
        if set(node_inputs) & set(node_outputs):
            # if overlap, no input node
            return True

    return False


def dump_pytorch_graph(graph):
    """
    List all the nodes in a PyTorch jit graph.
    Source: https://github.com/waleedka/hiddenlayer/blob/master/hiddenlayer/pytorch_builder.py
    """

    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))
    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))
