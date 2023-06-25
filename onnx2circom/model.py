# Ref: https://github.com/socathie/keras2circom/blob/main/keras2circom/model.py

from onnx import load
from onnx2circom.handler import transpile, print_circuit
import os

class Model:
    filename: str

    def __init__(self, filename: str):
        ''' Load a onnx model from a file. '''
        self.filename = filename
        
    def create_circuit(self, output_dir: str, verbose: bool, raw: bool):
        print(self.filename)

        onnx_model = load(self.filename)
        circuit = transpile(onnx_model)

        if raw:
            if circuit.components[-1].template.op_name == 'ArgMax':
                circuit.components.pop()  

        if verbose:
            print_circuit(circuit)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(os.path.join(output_dir, 'circuit.json'), 'w') as f:
            f.write(circuit.to_json())
            
        with open(os.path.join(output_dir, 'circuit.circom'), 'w') as f:
            f.write(circuit.to_circom())

