#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid for gradient calculation
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Forward pass
double forward(double inputs[], double weights[], double bias) {
    double z = 0.0;
    for (int i = 0; i < 2; i++) {
        z += inputs[i] * weights[i];
    }
    z += bias;
    return sigmoid(z);
}

// Training the perceptron using the Perceptron Learning Rule
void train(double weights[], double *bias, double learning_rate, int epochs) {
    // NAND gate truth table
    double training_inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double labels[4] = {1, 1, 1, 0};

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < 4; i++) {
            double *x = training_inputs[i];
            double y = labels[i];

            // Weighted sum
            double z = x[0] * weights[0] + x[1] * weights[1] + *bias;
            double output = sigmoid(z);
            double error = y - output;

            // Gradient descent update
            double delta = error * sigmoid_derivative(z);
            for (int j = 0; j < 2; j++) {
                weights[j] += learning_rate * delta * x[j];
            }
            *bias += learning_rate * delta;
        }
    }
}

int main() {
    double weights[2] = {0.0, 0.0};
    double bias = 0.0;
    double learning_rate = 0.1;
    int epochs = 10000;

    train(weights, &bias, learning_rate, epochs);

    printf("Trained weights: w1=%.4f, w2=%.4f\n", weights[0], weights[1]);
    printf("Trained bias: %.4f\n\n", bias);

    // Test the trained perceptron on NAND inputs
    int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (int i = 0; i < 4; i++) {
        double in[2] = {inputs[i][0], inputs[i][1]};
        double out = forward(in, weights, bias);
        printf("Input: %d %d -> Output: %.4f (rounded: %d)\n",
               inputs[i][0], inputs[i][1], out, out >= 0.5 ? 1 : 0);
    }

    return 0;
}
