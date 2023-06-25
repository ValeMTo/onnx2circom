pragma circom 2.0.0;

include "../keras2circom/node_modules/circomlib-ml/circuits/Conv2D.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/AveragePooling2D.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/Dense.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/ReLU.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/MaxPooling2D.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/Flatten2D.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/BatchNormalization2D.circom";
include "../keras2circom/node_modules/circomlib-ml/circuits/ArgMax.circom";

template Model() {
signal input in[32][32][3];
signal input conv2d_weights[2][2][3][30];
signal input conv2d_bias[30];
signal input conv2d_1_weights[2][2][30][10];
signal input conv2d_1_bias[10];
signal input dense_weights[10][8];
signal input dense_bias[8];
signal input batch_normalization_a[8];
signal input batch_normalization_b[8];
signal input dense_1_weights[8][12];
signal input dense_1_bias[12];
signal input dense_2_weights[12][4];
signal input dense_2_bias[4];
signal input batch_normalization_1_a[4];
signal input batch_normalization_1_b[4];
signal input dense_3_weights[784][1500];
signal input dense_3_bias[1500];
signal input dense_4_weights[1500][1200];
signal input dense_4_bias[1200];
signal input dense_5_weights[1200][4];
signal input dense_5_bias[4];
signal output out[1];

component conv2d = Conv2D(32, 32, 3, 30, 2, 1);
component max_pooling2d = MaxPooling2D(31, 31, 30, 1, 1);
component conv2d_1 = Conv2D(31, 31, 30, 10, 2, 2);
component dense = Dense(15, 15);
component dense_re_lu[15][15][8];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_re_lu[i0][i1][i2] = ReLU();
}}}
component batch_normalization = BatchNormalization2D(15, 15, 8);
component dense_1 = Dense(15, 15);
component dense_1_re_lu[15][15][12];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_1_re_lu[i0][i1][i2] = ReLU();
}}}
component dense_2 = Dense(15, 15);
component dense_2_re_lu[15][15][4];
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            dense_2_re_lu[i0][i1][i2] = ReLU();
}}}
component batch_normalization_1 = BatchNormalization2D(15, 15, 4);
component average_pooling2d = AveragePooling2D(15, 15, 4, 2, 1, 25000);
component flatten = Flatten2D(14, 14, 4);
component dense_3 = Dense(784, 1500);
component dense_3_re_lu[1500];
for (var i0 = 0; i0 < 1500; i0++) {
    dense_3_re_lu[i0] = ReLU();
}
component dense_4 = Dense(1500, 1200);
component dense_4_re_lu[1200];
for (var i0 = 0; i0 < 1200; i0++) {
    dense_4_re_lu[i0] = ReLU();
}
component dense_5 = Dense(1200, 4);
component dense_5_softmax = ArgMax(4);

for (var i0 = 0; i0 < 32; i0++) {
    for (var i1 = 0; i1 < 32; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            conv2d.in[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            for (var i3 = 0; i3 < 30; i3++) {
                conv2d.weights[i0][i1][i2][i3] <== conv2d_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 30; i0++) {
    conv2d.bias[i0] <== conv2d_bias[i0];
}
for (var i0 = 0; i0 < 31; i0++) {
    for (var i1 = 0; i1 < 31; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            max_pooling2d.in[i0][i1][i2] <== conv2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 31; i0++) {
    for (var i1 = 0; i1 < 31; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            conv2d_1.in[i0][i1][i2] <== max_pooling2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 2; i1++) {
        for (var i2 = 0; i2 < 30; i2++) {
            for (var i3 = 0; i3 < 10; i3++) {
                conv2d_1.weights[i0][i1][i2][i3] <== conv2d_1_weights[i0][i1][i2][i3];
}}}}
for (var i0 = 0; i0 < 10; i0++) {
    conv2d_1.bias[i0] <== conv2d_1_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 10; i2++) {
            dense.in[i0][i1][i2] <== conv2d_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 10; i0++) {
    for (var i1 = 0; i1 < 8; i1++) {
        dense.weights[i0][i1] <== dense_weights[i0][i1];
}}
for (var i0 = 0; i0 < 8; i0++) {
    dense.bias[i0] <== dense_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_re_lu[i0][i1][i2].in <== dense.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            batch_normalization.in[i0][i1][i2] <== dense_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 8; i0++) {
    batch_normalization.a[i0] <== batch_normalization_a[i0];
}
for (var i0 = 0; i0 < 8; i0++) {
    batch_normalization.b[i0] <== batch_normalization_b[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 8; i2++) {
            dense_1.in[i0][i1][i2] <== batch_normalization.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 8; i0++) {
    for (var i1 = 0; i1 < 12; i1++) {
        dense_1.weights[i0][i1] <== dense_1_weights[i0][i1];
}}
for (var i0 = 0; i0 < 12; i0++) {
    dense_1.bias[i0] <== dense_1_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_1_re_lu[i0][i1][i2].in <== dense_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 12; i2++) {
            dense_2.in[i0][i1][i2] <== dense_1_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 12; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        dense_2.weights[i0][i1] <== dense_2_weights[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    dense_2.bias[i0] <== dense_2_bias[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            dense_2_re_lu[i0][i1][i2].in <== dense_2.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            batch_normalization_1.in[i0][i1][i2] <== dense_2_re_lu[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    batch_normalization_1.a[i0] <== batch_normalization_1_a[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    batch_normalization_1.b[i0] <== batch_normalization_1_b[i0];
}
for (var i0 = 0; i0 < 15; i0++) {
    for (var i1 = 0; i1 < 15; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            average_pooling2d.in[i0][i1][i2] <== batch_normalization_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 14; i0++) {
    for (var i1 = 0; i1 < 14; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            flatten.in[i0][i1][i2] <== average_pooling2d.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 784; i0++) {
    dense_3.in[i0] <== flatten.out[i0];
}
for (var i0 = 0; i0 < 784; i0++) {
    for (var i1 = 0; i1 < 1500; i1++) {
        dense_3.weights[i0][i1] <== dense_3_weights[i0][i1];
}}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_3.bias[i0] <== dense_3_bias[i0];
}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_3_re_lu[i0].in <== dense_3.out[i0];
}
for (var i0 = 0; i0 < 1500; i0++) {
    dense_4.in[i0] <== dense_3_re_lu[i0].out;
}
for (var i0 = 0; i0 < 1500; i0++) {
    for (var i1 = 0; i1 < 1200; i1++) {
        dense_4.weights[i0][i1] <== dense_4_weights[i0][i1];
}}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_4.bias[i0] <== dense_4_bias[i0];
}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_4_re_lu[i0].in <== dense_4.out[i0];
}
for (var i0 = 0; i0 < 1200; i0++) {
    dense_5.in[i0] <== dense_4_re_lu[i0].out;
}
for (var i0 = 0; i0 < 1200; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        dense_5.weights[i0][i1] <== dense_5_weights[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    dense_5.bias[i0] <== dense_5_bias[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    dense_5_softmax.in[i0] <== dense_5.out[i0];
}
out[0] <== dense_5_softmax.out;

}

component main = Model();
