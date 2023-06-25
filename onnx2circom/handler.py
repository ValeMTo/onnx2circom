# Ref: https://github.com/socathie/keras2circom/blob/main/keras2circom/transpiler.py

from keras2circom.keras2circom.circom import Component, Signal, Template 
from keras2circom.keras2circom.circom import dir_parse, templates
from onnx.onnx_ml_pb2 import NodeProto, ModelProto
from onnx import numpy_helper
from keras2circom.keras2circom.circom import Circuit, templates
import numpy as np

supported_ops = [
    'Conv',
    'BatchNormalization',
    'MatMul',
    'Relu',
    'Reshape', #Flatten
    'AveragePool',
    'MaxPool',
    'ReduceMax', # Global Max Pooling
    'ReduceMean', # Global Average Pooling
    'GlobalMaxPool',
    'GlobalAveragePool',
    'Softmax'
]

skip_ops =  [
    'Unsqueeze',
    'Shape',
    'Cast',
    'Slice',
    'Concat',
    'Transpose',
    'Squeeze',
    'Add',
    'Gather',
    'ReduceProd', 
]


def init():
    dir_parse('../keras2circom/node_modules/circomlib-ml/circuits/', skips=['util.circom', 'circomlib-matrix', 'circomlib', 'crypto'])

def check_available_ops(op: str, name: str) -> bool:
    ''' Check if the operation is supported by circom. '''
    if op in supported_ops:
        if op == 'Reshape':
            if 'flatten' in name.lower():
                return True
            return False
        if op == 'ReduceMax':
            if 'global_max_pooling' in name.lower():
                return True
        if op == 'ReduceMean':
            if 'global_average_pooling' in name.lower():
                return True
            return False
        return True
    elif op in skip_ops:
        return False
    
    raise NotImplementedError(f'Unsupported op: {op} - Circom component not found.')

def transpile(model: ModelProto):
    ''' Transpile a onnx model to a CIRCOM circuit.'''
    init()
    circuit = Circuit()
        
    input_shape = calculate_first_input(model.graph.input[0])

    all_weights = get_weights(model)

    for index, node in enumerate(model.graph.node):
        if check_available_ops(node.op_type, node.name):
            node_weights = get_layer_weights(extract_node_name(node), all_weights)
            output_shape = calculate_output(input_shape, node, node_weights, model)
            component = transpile_node(input_shape, output_shape, node, node_weights)

            circuit.add_component(component)
            input_shape = output_shape
        

    
    return circuit

def transpile_node(input_shape: tuple, output_shape: tuple, node: NodeProto, weights: dict,):
    '''  Transpile a node to a CIRCOM component.'''
    if len(extract_node_name(node).split('/'))>1:
        node_name = extract_node_name(node).split('/')[1]
    else:
        node_name = extract_node_name(node)

    if node.op_type == 'Conv':
        kernel_shape = get_kernel_shape(node)
        strides_shape = get_strides_shape(node)
        padding = get_padding(node)
        filters = get_filters(weights)
        dilation  = get_dilation_rate(node)
        group = get_groups(node)
        return transpile_Conv(node_name, input_shape, output_shape, weights, kernel_shape, strides_shape, filters,  padding, dilation, group)
    elif node.op_type == "BatchNormalization":
        epsilon = get_epsilon(node)
        return transpile_BatchNormalization(node_name, input_shape, output_shape, weights, epsilon)
    elif node.op_type == "MatMul":
        return transpile_MatMul(node_name, input_shape, output_shape, weights)
    elif node.op_type == "Relu":
        return transpile_Relu(node_name, input_shape, output_shape)
    elif node.op_type == "AveragePool":
        kernel_shape = get_kernel_shape(node)
        strides_shape = get_strides_shape(node)
        padding = get_padding(node)
        return transpile_AveragePool(node_name, input_shape, output_shape, kernel_shape, strides_shape, padding)
    elif node.op_type == "MaxPool":
        kernel_shape = get_kernel_shape(node)
        strides_shape = get_strides_shape(node)
        padding = get_padding(node)
        return transpile_MaxPool(node_name, input_shape, output_shape, kernel_shape, strides_shape, padding)
    elif node.op_type == "Reshape": # only Flatten
        return transpile_Flatten(node_name, input_shape, output_shape)
    elif node.op_type == "GlobalMaxPool" or (node.op_type == "ReduceMax" and 'global_max_pooling' in node.name.lower()):
        return transpile_GlobalMaxPool(node_name, input_shape, output_shape)
    elif node.op_type == "GlobalAveragePool" or (node.op_type == "ReduceMean" and 'global_average_pooling' in node.name.lower()):
        return transpile_GlobalAveragePool(node_name, input_shape, output_shape)
    elif node.op_type == "Softmax":
        return transpile_Softmax(node_name, input_shape, output_shape)

def calculate_first_input(model_input): #model.graph.input[0]
    ''' Calculate the input shape of the first layer. '''
    input_shape = [dim.dim_value for dim in model_input.type.tensor_type.shape.dim][1:]
    return tuple(input_shape)

def extract_node_name(node: NodeProto):
    ''' Extract the name of a node: model/layer, when model name is available '''
    output_name = str(node.output[0])
    output_names = output_name.split('/')
    output_name = '/'.join(output_names[:2])
    return output_name

def get_padding(node: NodeProto):
    ''' Get the padding type of a node.'''
    padding = 'valid'
    for attribute in node.attribute:
        if 'pad' in attribute.name.lower():
            if 'same' in attribute.s:
                padding = 'same'
    return padding
  
def get_dilation_rate(node: NodeProto):
    ''' Get the dilation rate of a node.'''
    for attribute in node.attribute:
        if attribute.name.lower() == 'dilations':
            dilatations = (attribute.ints[0], attribute.ints[1])
            return dilatations
        
    raise AttributeError('Dilatations not found')

def get_groups(node: NodeProto):
    ''' Get the groups of a node.'''
    for attribute in node.attribute:
        if attribute.name.lower() == 'group':
            groups = attribute.i
            return groups
        
    raise AttributeError('Groups not found')

def get_kernel_shape(node: NodeProto):
    ''' Get the kernel shape of a node.'''
    for attribute in node.attribute:
        if attribute.name == 'kernel_shape':
            kernel_shape = (attribute.ints[0], attribute.ints[1])
            return kernel_shape
        
    raise AttributeError('Kernel shape not found')
    
def get_strides_shape(node: NodeProto):
    ''' Get the strides shape of a node.'''
    for attribute in node.attribute:
        if attribute.name.lower() == 'strides':
            stride_shape = (attribute.ints[0], attribute.ints[1])
            return stride_shape
        
    raise AttributeError('Stride shape not found')

def get_filters(weights: dict):
    ''' Get the number of filters of a node.'''
    for key, value in weights.items():
        if 'bias' not in key.lower():
            nChannels = value.shape[-1]
            return nChannels


    


    raise AttributeError('Number of channels not available')
    

def calculate_output(input_shape: tuple, node: NodeProto, weights: dict, model: ModelProto):
    ''' Calculate the output shape of a node. '''
    if node.op_type == "Conv":
        
        input_height, input_width = input_shape[:-1]
        filter_height, filter_width = get_kernel_shape(node)
        strides = get_strides_shape(node)[0]
        nChannels = get_filters(weights)

        # Padding always valid: no padding
        output_height = (input_height - filter_height) / strides + 1
        output_width = (input_width - filter_width) / strides + 1

        return (int(output_height), int(output_width), int(nChannels))
    
    elif node.op_type == "MatMul":
        output_name = extract_node_name(node)
        for value_info in model.graph.initializer:
            if output_name in value_info.name:
                last = value_info
        num_neurons = last.dims[0]
        return input_shape[:-1] + (num_neurons,)
    
    elif node.op_type == "Reshape":
        if 'flatten' not in node.name.lower():
            raise AttributeError('Only flatten is supported')
        
        flatten_dim = 1
        for dim in input_shape:
            flatten_dim *= dim

        return (flatten_dim,)
    
    elif node.op_type == "AveragePool" or node.op_type == "MaxPool":
        input_height, input_width, input_channels = input_shape
        pool_height, pool_width = get_kernel_shape(node)
        stride_height, stride_width = get_strides_shape(node)

        output_height = (input_height - pool_height) / stride_height + 1
        output_width = (input_width - pool_width) / stride_width + 1
        output_channels = input_channels

        return (int(output_height), int(output_width), int(output_channels))
    
    elif node.op_type == "GlobalMaxPool" or node.op_type == "GlobalAveragePool" or node.op_type == "ReduceMax" or node.op_type == "ReduceMean":
        return (input_shape[-1], )

    return input_shape

def get_epsilon(node: NodeProto):
    ''' Get the epsilon of a node.'''
    for attribute in node.attribute:
        if attribute.name == 'epsilon':
            return attribute.f

    raise AttributeError('Epsilon not found')

def get_weights(model: ModelProto):
    ''' Get all weights of a model. '''
    onnx_weights = model.graph.initializer
    weights = {}
    for onnx_w in onnx_weights:
        try:
            if len(onnx_w.ListFields()) < 4:
                onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
            else:
                onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)
        except:
            onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
            weights[onnx_extracted_weights_name] = numpy_helper.to_array(onnx_w)

    weights_global = {key: value for key, value in weights.items() if 'ReadVariableOp' in key}
    return weights_global

def get_layer_weights(name: str, weights: dict):
    ''' Get the weights of a layer. Return {} if no weights are found.'''

    layer_weights = {}

    for key, value in weights.items():
        if name in key:
            layer_weights[key] = value

    for key, value in layer_weights.items():
        if 'Conv2D' in key:
           layer_weights[key] = np.transpose(layer_weights[key] , (2, 3, 1, 0))

    layer_weights =  dict(sorted(layer_weights.items()))
        
    return layer_weights


def transpile_Conv(node_name: str, input_shape: tuple, output_shape: tuple, weights: dict, kernel_size: tuple, strides: tuple, filters: int, padding: str, dilation_rate: tuple, groups: int):
    ''' Transpile a Conv2D layer. '''
    if padding != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if strides[0] != strides[1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')
    if kernel_size[0] != kernel_size[1]:
        raise NotImplementedError('Only kernel_size[0] == kernel_size[1] is supported')
    if dilation_rate[0] != 1:
        raise NotImplementedError('Only dilation_rate[0] == 1 is supported')
    if dilation_rate[1] != 1:
        raise NotImplementedError('Only dilation_rate[1] == 1 is supported')
    if groups != 1:
        raise NotImplementedError('Only groups == 1 is supported')
    
    '''
    if layer.config['activation'] not in ['linear', 'relu']:
        raise NotImplementedError(f'Activation {layer.config["activation"]} is not supported')
    
    if layer.config['use_bias'] == False:
        layer.weights.append(np.zeros(layer.weights[0].shape[-1]))
    '''

    bias = np.empty(0)
    for key, value in weights.items():
        if not 'bias' in key.lower():
            weights = value
        else:
            bias = value

    if len(bias) == 0:
        bias = np.zeros(weights[0].shape[-1])

    conv = Component(node_name, templates['Conv2D'], [
        Signal('in', input_shape),
        Signal('weights', weights.shape, weights),
        Signal('bias', bias.shape, bias),
        ],[Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        'nFilters': filters,
        'kernelSize': kernel_size[0],
        'strides': strides[0],
        })
    
    #if layer.config['activation'] == 'relu':
    #    activation = Component(layer.name+'_re_lu', templates['ReLU'], [Signal('in', layer.output)], [Signal('out', layer.output)])
    #    return [conv, activation]
    
    return conv

def transpile_BatchNormalization(node_name: str, input_shape: tuple, output_shape: tuple, weights: dict, epsilon: float):
    ''' Transpile a BatchNormalization2D layer. '''
    if len(input_shape) != 3:
        raise NotImplementedError('Only 2D inputs are supported')
    if len(weights) != 4:
        raise NotImplementedError('Only center=True and scale=True supported')

    weights = list(weights.values())

    moving_mean = weights[0]
    moving_var = weights[1]

    gamma = weights[2]
    beta = weights[3]

    a = gamma/(moving_var+epsilon)**.5
    b = beta-gamma*moving_mean/(moving_var+epsilon)**.5
    
    batch_normalisation = Component(node_name, templates['BatchNormalization2D'], [
        Signal('in', input_shape),
        Signal('a', a.shape, a),
        Signal('b', b.shape, b),
        ],[Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        })
    return batch_normalisation


def transpile_AveragePool(node_name: str, input_shape: tuple, output_shape: tuple, pool_size: tuple, strides: tuple, padding: str):
    ''' Transpile a AveragePooling2D layer.'''
    if padding != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if pool_size[0] != pool_size[1]:
        raise NotImplementedError('Only pool_size[0] == pool_size[1] is supported')
    if strides[0] != strides[1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')

    avg_pooling = Component(node_name, templates['AveragePooling2D'], 
        [Signal('in', input_shape)], 
        [Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        'poolSize': pool_size[0],
        'strides': strides[0],
        'scaledInvPoolSize': 1/(pool_size[0]**2),
        })
    
    return avg_pooling

def transpile_MaxPool(node_name: str, input_shape: tuple, output_shape: tuple, pool_size: tuple, strides: tuple, padding: str):
    ''' Transpile a MaxPooling2D layer.'''
    if padding != 'valid':
        raise NotImplementedError('Only padding="valid" is supported')
    if pool_size[0] != pool_size[1]:
        raise NotImplementedError('Only pool_size[0] == pool_size[1] is supported')
    if strides[0] != strides[1]:
        raise NotImplementedError('Only strides[0] == strides[1] is supported')

    max_pooling = Component(node_name, templates['MaxPooling2D'], 
        [Signal('in', input_shape)], 
        [Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        'poolSize': pool_size[0],
        'strides': strides[0],
        })
    
    return max_pooling

def transpile_GlobalMaxPool(node_name: str, input_shape: tuple, output_shape: tuple):
    ''' Transpile a GlobalMaxPooling2D layer.'''
    global_max_pooling = Component(node_name, templates['GlobalMaxPooling2D'], [
        Signal('in', input_shape),
        ],[Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        })

    return global_max_pooling

def transpile_GlobalAveragePool(node_name: str, input_shape: tuple, output_shape: tuple):
    ''' Transpile a GlobalAveragePooling2D layer.'''
    global_average_pooling = Component(node_name, 
        templates['GlobalAveragePooling2D'], [
        Signal('in', input_shape),
        ],[Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        'scaledInv': 1/(input_shape[0]*input_shape[1]),
        })

    return global_average_pooling

def transpile_Softmax(node_name: str, input_shape: tuple, output_shape: tuple):
    ''' Transpile a Softmax layer.'''
    softmax = Component(node_name+'_softmax', 
        templates['ArgMax'], 
        [Signal('in', input_shape)], 
        [Signal('out', (1,))], 
        {'n': output_shape[0]})

    return softmax

def transpile_MatMul(node_name: str, input_shape: tuple, output_shape: tuple, weights: dict):
    ''' Transpile a Dense layer.'''
    if len(weights) != 2:
        raise AttributeError('Number of weights arrays is not equal 2 for the dense layer')

    for key, value in weights.items():
        if not 'bias' in key.lower():
            weights = value
        else:
            bias = value

    if len(bias) == 0:
        bias = np.zeros(weights[0].shape[-1])

    dense = Component(node_name, templates['Dense'], [
        Signal('in', input_shape),
        Signal('weights', weights.shape, weights),
        Signal('bias', bias.shape, bias),
        ],[Signal('out', output_shape )],{
        'nInputs':  input_shape[0],
        'nOutputs': output_shape[0],
        })
    
    return dense

def transpile_Relu(node_name: str, input_shape: tuple, output_shape: tuple):
    ''' Transpile a ReLU layer.'''
    activation = Component(node_name+'_re_lu', templates['ReLU'], 
        [Signal('in', input_shape)], 
        [Signal('out', output_shape)])
    
    return activation

def transpile_Flatten(node_name: str, input_shape: tuple, output_shape: tuple):
    ''' Transpile a Flatten layer.'''
    if len(input_shape) != 3:
        raise NotImplementedError('Only 2D inputs are supported')

    activation = Component(node_name, templates['Flatten2D'], [
        Signal('in', input_shape),
        ],[Signal('out', output_shape)],{
        'nRows': input_shape[0],
        'nCols': input_shape[1],
        'nChannels': input_shape[2],
        })
    
    return activation

def print_circuit(circuit: Circuit):
    for component in circuit.components:
        print('Transpiled layer: ',  component.name)
        for signal in component.inputs:
            print('input:', signal.name, signal.shape)
        for signal in component.outputs:
            print('output:', signal.name, signal.shape)
        print('\n-------------------------------------------------')