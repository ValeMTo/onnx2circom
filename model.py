# Ref: https://github.com/socathie/keras2circom/blob/main/keras2circom/model.py

from onnx import load
from onnx.onnx_ml_pb2 import ModelProto
from keras2circom.keras2circom.circom import Circuit
import handler

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

skip_ops =  [
    'Unsqueeze',
    'Shape',
    'Cast',
    'Slice',
    'Concat',
    'Reshape',
    'Transpose',
    'Squeeze',
    'Add',
    'Gather',
    'ReduceProd'
]

class Model:
    onnx_model: ModelProto

    def __init__(self, filename: str):
        ''' Load a onnx model from a file. '''
        self.model = load(filename)
    
    def check_available_ops(op: str) -> bool:
        if op in supported_ops:
            return True
        elif op in skip_ops:
            return False
        raise NotImplementedError(f'Unsupported op: {op}')

    def create_circuit(self):
        circuit = Circuit()

        input_shape = handler.calculate_first_input(self.model.graph.input)

        for index, node in enumerate(self.onnx_model.graph.node):
            if Model.check_available_ops(node.op_type):

                component, input_shape = handler.transpile_node(input_shape, index, self.model)

                circuit.add_component(component)

        return circuit