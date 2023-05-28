# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing
import onnx
import numpy as np

supported_ops = [
    "Conv",
    "BatchNormalization",
    "MatMul",
    "Relu",
    "AveragePool",
    "MaxPool",
    "GlobalMaxPool",
    "GlobalAveragePool",
    "Softmax"
]


# read each node in onnx model and convert it to a class called Layer, if the node is useful
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    #input: typing.List[int]
    #output: typing.List[int]
    #config: typing.Dict[str, typing.Any]
    #weights: typing.List[np.ndarray]

    def __init__(self, node: onnx.onnx_ml_pb2.NodeProto):
        self.op = node.op_type
        self.name = node.name
        #self.input = layer.input_shape[1:]
        #self.output = layer.output_shape[1:]
        #self.config = layer.get_config()
        #self.weights = layer.get_weights()

class Model:
    layers: typing.List[Layer]

    def __init__(self, filename: str, raw: bool = False):
        ''' Load a Keras model from a file. '''
        model = onnx.load(filename)
        
        network_layers = []
        previous_node = model.graph.node[0]
        for node in model.graph.node[1:]:
            if node.op_type in supported_ops:
                if previous_node.op_type =='Relu' and node.op_type.equals =='MatMul':
                    network_layers.pop()
                network_layers.append(node)

        self.layers = [Layer(node) for node in network_layers if self._for_transpilation(node.op_type)]
    
    @staticmethod
    def _for_transpilation(op: str) -> bool:
        if op in supported_ops:
            return True
        raise NotImplementedError(f'Unsupported op: {op}')