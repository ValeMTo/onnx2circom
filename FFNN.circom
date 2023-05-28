/* 
Compile with circom file_name.circom --r1cs --wasm --sym --c

Prendi una rete piccola, provare a scrivere il circum-ml e
capirne la difficoltà (fare lo stesso con Keras2Circum)
*/

/*
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    hidden_layer1 = tfkl.Dense(units=64, activation='relu', name='Hidden1')(input_layer)
    hidden_layer2 = tfkl.Dense(units=64, activation='relu', name='Hidden2')(hidden_layer1)
    output_layer = tfkl.Dense(units=1, activation='linear', name='Output')(hidden_layer2)

    CIRCOM signals of DENSE component

    signal input in[nInputs];
    signal input weights[nInputs][nOutputs];
    signal input bias[nOutputs];
    signal output out[nOutputs];
*/

pragma circom 2.1.4

include "./circomlib-ml-master/circuits/ArgMax.circom"
include "./circomlib-ml-master/circuits/AveragePooling2D.circom"
include "./circomlib-ml-master/circuits/BatchNormalization2D.circom"
include "./circomlib-ml-master/circuits/Conv1D.circom"
include "./circomlib-ml-master/circuits/Conv2D.circom"
include "./circomlib-ml-master/circuits/Dense.circom"
include "./circomlib-ml-master/circuits/Flatten2D.circom"
include "./circomlib-ml-master/circuits/GlobalAveragePooling2D.circom"
include "./circomlib-ml-master/circuits/GlobalMaxPooling2D.circom"
include "./circomlib-ml-master/circuits/GlobalSumPooling2D.circom"
include "./circomlib-ml-master/circuits/MaxPooling2D.circom"
include "./circomlib-ml-master/circuits/ReLU.circom"
include "./circomlib-ml-master/circuits/SumPooling2D.circom"

template NeuralNetwork (nInputs) {  

    // Declaration of signals.  
    signal input a[nInputs];  
    Dense hidden1(13, 64);
    Dense hidden2(64, 64);
    Dense output(64, 1);
 
    hidden1.in
 }

 
 component main = NeuralNetwork();
