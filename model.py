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

needed_next = [
    "MatMul",
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
    
    def create_circuit():
        circuit = Circuit()
        previous_node = ''
        for index, node in enumerate(self.onnx_model.graph.node):
            if previous_node not in needed_next:
                if check_available_ops(node.op_type):
                        if node.op_type in needed_next:
                            circuit.add_component(handler.transpile_two_nodes(node, self.onnx_model.graph.node[index + 1]))

                        else:
                            circuit.add_component(handler.transpile_node(node))
            previous_node = node.op_type
        return circuit