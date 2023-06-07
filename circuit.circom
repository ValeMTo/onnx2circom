pragma circom 2.0.0;

include "../keras2circom/node_modules/circomlib-ml/circuits/Dense.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/ReLU.circom";

template Model() {
signal input in[32][32][3];
signal input dense_3_weights[3][8];
signal input dense_3_bias[8];
signal output out[32][32][8];

component dense_3 = Dense(32, 32);
component dense_3_re_lu[32][32][8];
for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_3_re_lu[i0][i1][i2] = ReLU();
}}}

for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            dense_3.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        dense_3.weights[i0][i1] <== dense_3_weights[i0][i1];
}}
for (var i0 = 0; i0 < 8; i0++) {
    dense_3.bias[i0] <== dense_3_bias[i0];
}
for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_3_re_lu[i0][i1][i2].in <== dense_3.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            out[i0][i1][i2] <== dense_3_re_lu[i0][i1][i2].out;
}}}

}

component main = Model();
