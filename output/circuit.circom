pragma circom 2.0.0;

include "../../keras2circom/node_modules/circomlib-ml/circuits/Flatten2D.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/ReLU.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/BatchNormalization2D.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/Dense.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/Conv2D.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/MaxPooling2D.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/ArgMax.circom";
include "../../keras2circom/node_modules/circomlib-ml/circuits/AveragePooling2D.circom";

template Model() {
signal input in[32][32][3];
signal input conv2d_9_weights[2][2][3][30];
signal input conv2d_9_bias[30];
signal input conv2d_10_weights[2][2][30][10];
signal input conv2d_10_bias[10];
signal input dense_22_weights[10][8];
signal input dense_22_bias[8];
signal input batch_normalization_8_a[8];
signal input batch_normalization_8_b[8];
signal input dense_23_weights[8][12];
signal input dense_23_bias[12];
signal input dense_24_weights[12][4];
signal input dense_24_bias[4];
signal input batch_normalization_9_a[4];
signal input batch_normalization_9_b[4];
signal input dense_25_weights[784][1500];
signal input dense_25_bias[1500];
signal input dense_26_weights[1500][1200];
signal input dense_26_bias[1200];
signal input dense_27_weights[1200][4];
signal input dense_27_bias[4];
signal output out[1];

component conv2d_9 = Conv2D(32, 32, 3, 30, 2, 1);
component max_pooling2d_3 = MaxPooling2D(31, 31, 30, 1, 1);
component conv2d_10 = Conv2D(31, 31, 30, 10, 2, 2);
component dense_22 = Dense(15, 15);
component dense_22_re_lu[15][15][8];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_22_re_lu[i0][i1][i2] = ReLU();
}}}
component batch_normalization_8 = BatchNormalization2D(15, 15, 8);
component dense_23 = Dense(15, 15);
component dense_23_re_lu[15][15][12];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_23_re_lu[i0][i1][i2] = ReLU();
}}}
component dense_24 = Dense(15, 15);
component dense_24_re_lu[15][15][4];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            dense_24_re_lu[i0][i1][i2] = ReLU();
}}}
component batch_normalization_9 = BatchNormalization2D(15, 15, 4);
component average_pooling2d_3 = AveragePooling2D(15, 15, 4, 2, 1, 25000);
component flatten_2 = Flatten2D(14, 14, 4);
component dense_25 = Dense(784, 1500);
component dense_25_re_lu[1500];
for (var i0 = 0; i0 < 1500; i0++) {
    dense_25_re_lu[i0] = ReLU();
}
component dense_26 = Dense(1500, 1200);
component dense_26_re_lu[1200];
for (var i0 = 0; i0 < 1200; i0++) {
    dense_26_re_lu[i0] = ReLU();
}
component dense_27 = Dense(1200, 4);
component dense_27_softmax = ArgMax(4);

for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            conv2d_9.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            for (var i3 = 0; i3 < 30; i3++) {
                conv2d_9.weights[i0][i1][i2][i3] <== conv2d_9_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 30; i0++) {
    conv2d_9.bias[i0] <== conv2d_9_bias[i0];
}
for (var i0 = 0; i0 < 31; i0++) {
    for (var i1 = 0; i1 < 31; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            max_pooling2d_3.in[i0][i1][i2] <== conv2d_9.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 31; i0++) {
    for (var i1 = 0; i1 < 31; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            conv2d_10.in[i0][i1][i2] <== max_pooling2d_3.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            for (var i3 = 0; i3 < 10; i3++) {
                conv2d_10.weights[i0][i1][i2][i3] <== conv2d_10_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 10; i0++) {
    conv2d_10.bias[i0] <== conv2d_10_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 10; i2++) {
            dense_22.in[i0][i1][i2] <== conv2d_10.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 10; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        dense_22.weights[i0][i1] <== dense_22_weights[i0][i1];
}}
for (var i0 = 0; i0 < 8; i0++) {
    dense_22.bias[i0] <== dense_22_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_22_re_lu[i0][i1][i2].in <== dense_22.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            batch_normalization_8.in[i0][i1][i2] <== dense_22_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 8; i0++) {
    batch_normalization_8.a[i0] <== batch_normalization_8_a[i0];
}
for (var i0 = 0; i0 < 8; i0++) {
    batch_normalization_8.b[i0] <== batch_normalization_8_b[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_23.in[i0][i1][i2] <== batch_normalization_8.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        dense_23.weights[i0][i1] <== dense_23_weights[i0][i1];
}}
for (var i0 = 0; i0 < 12; i0++) {
    dense_23.bias[i0] <== dense_23_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_23_re_lu[i0][i1][i2].in <== dense_23.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_24.in[i0][i1][i2] <== dense_23_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        dense_24.weights[i0][i1] <== dense_24_weights[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    dense_24.bias[i0] <== dense_24_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            dense_24_re_lu[i0][i1][i2].in <== dense_24.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            batch_normalization_9.in[i0][i1][i2] <== dense_24_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    batch_normalization_9.a[i0] <== batch_normalization_9_a[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    batch_normalization_9.b[i0] <== batch_normalization_9_b[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            average_pooling2d_3.in[i0][i1][i2] <== batch_normalization_9.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 14; i0++) {
    for (var i1 = 0; i1 < 14; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            flatten_2.in[i0][i1][i2] <== average_pooling2d_3.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 784; i0++) {
    dense_25.in[i0] <== flatten_2.out[i0];
}
for (var i0 = 0; i0 < 784; i0++) {
    for (var i1 = 0; i1 < 1500; i1++) {
        dense_25.weights[i0][i1] <== dense_25_weights[i0][i1];
}}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_25.bias[i0] <== dense_25_bias[i0];
}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_25_re_lu[i0].in <== dense_25.out[i0];
}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_26.in[i0] <== dense_25_re_lu[i0].out;
}
for (var i0 = 0; i0 < 1500; i0++) {
    for (var i1 = 0; i1 < 1200; i1++) {
        dense_26.weights[i0][i1] <== dense_26_weights[i0][i1];
}}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_26.bias[i0] <== dense_26_bias[i0];
}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_26_re_lu[i0].in <== dense_26.out[i0];
}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_27.in[i0] <== dense_26_re_lu[i0].out;
}
for (var i0 = 0; i0 < 1200; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        dense_27.weights[i0][i1] <== dense_27_weights[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    dense_27.bias[i0] <== dense_27_bias[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    dense_27_softmax.in[i0] <== dense_27.out[i0];
}
out[0] <== dense_27_softmax.out;

}

component main = Model();
