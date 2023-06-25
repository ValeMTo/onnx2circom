# Ref: https://github.com/socathie/keras2circom/blob/main/keras2circom/model.py

from onnx import load
from handler import transpile

class Model:
    filename: str

    def __init__(self, filename: str):
        ''' Load a onnx model from a file. '''
        self.filename = filename
        
    def create_circuit(self):

        print(self.filename)

        onnx_model = load(self.filename)
        circuit = transpile(onnx_model)
        
        return circuit