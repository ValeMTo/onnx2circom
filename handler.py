# Ref: https://github.com/socathie/keras2circom/blob/main/keras2circom/transpiler.py

from keras2circom.keras2circom.circom import Component, Signal, Template 
from keras2circom.keras2circom.circom import dir_parse, templates
from onnx.onnx_ml_pb2 import NodeProto, ModelProto


class Handler:

    def __init__():
        dir_parse('keras2circom/node_modules/circomlib-ml/circuits/', skips=['util.circom', 'circomlib-matrix', 'circomlib', 'crypto'])

    def transpile_node(input_shape: tuple, index: int, model: ModelProto):
        '''  Transpile a node to a CIRCOM component.'''
        node = model.graph.node[index]
        output_shape = Handler.calculate_output(input_shape, index, model)
    
        if node.op_type == 'Conv':
            return Handler.transpile_Conv(node, input_shape, output_shape)
        elif node.op_type == "BatchNormalization":
            return Handler.transpile_BatchNormalization(node, input_shape, output_shape)
        elif node.op_type == "MatMul":
            return Handler.transpile_MatMul(node, input_shape, output_shape)
        elif node.op_type == "Relu":
            return Handler.transpile_Relu(node, input_shape, output_shape)
        elif node.op_type == "AveragePool":
            return Handler.transpile_AveragePool(node, input_shape, output_shape)
        elif node.op_type == "MaxPool":
            return Handler.transpile_MaxPool(node, input_shape, output_shape)
        elif node.op_type == "GlobalMaxPool":
            return Handler.transpile_GlobalMaxPool(node, input_shape, output_shape)
        elif node.op_type == "GlobalAveragePool":
            return Handler.transpile_GlobalAveragePool(node, input_shape, output_shape)
        elif node.op_type == "Softmax":
            return Handler.transpile_Softmax(node, input_shape, output_shape)

    def calculate_first_input(model_input): #model.graph.input[0]
        input_shape = [dim.dim_value for dim in model_input.type.tensor_type.shape.dim][1:]
        return tuple(input_shape)
    
    def extract_node_name(node: NodeProto):
        output_name = str(node.output[0])
        output_names = output_name.split('/')
        output_name = '/'.join(output_names[:2])
        return output_name

    def calculate_output(input_shape: tuple, index: int, model: ModelProto):
        node = model.graph.node[index]

        if node.op_type == "Conv":
            # Extract filter dimension
            for attribute in node.attribute:
                if attribute.name == 'kernel_shape':
                    kernel_shape =  [attribute.ints[1], attribute.ints[1]]
                    print('Kernel shape: ', kernel_shape)
            
            # Extract number of filters
            # Complete
            raise NotImplementedError(' Started, not completed')
        
        elif node.op_type == "MatMul":
            output_name = Handler.extract_node_name(node)
            for value_info in model.graph.initializer:
                if output_name in value_info.name:
                    last = value_info
                    print(value_info)
            num_neurons = last.dims
            print(f"The number of neurons in the node is {num_neurons}")
            return input_shape[:-1] + (num_neurons,)
        
        elif node.op_type == "AveragePool":
            raise NotImplementedError()
        elif node.op_type == "MaxPool":
            raise NotImplementedError()
        elif node.op_type == "GlobalMaxPool":
            raise NotImplementedError()
        elif node.op_type == "GlobalAveragePool":
            raise NotImplementedError()

        return input_shape

    def transpile_Conv(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_BatchNormalization(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_AveragePool(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_MaxPool(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_GlobalMaxPool(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_GlobalAveragePool(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()

    def transpile_Softmax(node: NodeProto, input_shape: tuple, output_shape: tuple):
        raise NotImplementedError()
    
    def transpile_MatMul(node: NodeProto, input_shape: tuple, output_shape: tuple):
        #Fix weights

        dense = Component(Handler.extract_node_name(node).split()[1], templates['Dense'], [
            Signal('in', input_shape),
            Signal('weights', layer_1.weights[0].shape, layer_1.weights[0]),
            Signal('bias', layer_1.weights[1].shape, layer_1.weights[1]),
            ],[Signal('out', )],{
            'nInputs':  input_shape[0],
            'nOutputs': output_shape[0],
            })
        
        return dense
    
    def transpile_Relu(node: NodeProto, input_shape: tuple, output_shape: tuple):
        activation = Component(Handler.extract_node_name(node).split()[1], templates['ReLU'], 
            [Signal('in', input_shape)], 
            [Signal('out', output_shape)])
        
        return activation
