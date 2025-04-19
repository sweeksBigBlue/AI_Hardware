#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LEARNING_RATE 0.5
#define EPOCHS 10000

// Sigmoid + derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// XOR data
double inputs[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

double targets[4] = {0, 1, 1, 0};

int main() {
    srand(time(NULL));

    // Weights and biases
    double w_input_hidden[2][2];  // 2 inputs → 2 hidden
    double b_hidden[2];
    double w_hidden_output[2];    // 2 hidden → 1 output
    double b_output;

    // Random initialization [-1, 1]
    for (int i = 0; i < 2; i++) {
        b_hidden[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        w_hidden_output[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 2; j++) {
            w_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    b_output = ((double)rand() / RAND_MAX) * 2 - 1;

    // Training
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int sample = 0; sample < 4; sample++) {
            double x1 = inputs[sample][0];
            double x2 = inputs[sample][1];
            double y = targets[sample];

            // Forward pass
            double h_in[2], h_out[2];
            for (int i = 0; i < 2; i++) {
                h_in[i] = x1 * w_input_hidden[0][i] + x2 * w_input_hidden[1][i] + b_hidden[i];
                h_out[i] = sigmoid(h_in[i]);
            }

            double z = h_out[0] * w_hidden_output[0] + h_out[1] * w_hidden_output[1] + b_output;
            double output = sigmoid(z);

            // Backpropagation
            double error = y - output;
            double d_output = error * sigmoid_derivative(z);

            double d_hidden[2];
            for (int i = 0; i < 2; i++) {
                d_hidden[i] = d_output * w_hidden_output[i] * sigmoid_derivative(h_in[i]);
            }

            // Update output weights and bias
            for (int i = 0; i < 2; i++) {
                w_hidden_output[i] += LEARNING_RATE * d_output * h_out[i];
            }
            b_output += LEARNING_RATE * d_output;

            // Update input→hidden weights and biases
            for (int i = 0; i < 2; i++) {
                w_input_hidden[0][i] += LEARNING_RATE * d_hidden[i] * x1;
                w_input_hidden[1][i] += LEARNING_RATE * d_hidden[i] * x2;
                b_hidden[i] += LEARNING_RATE * d_hidden[i];
            }
        }
    }

    // Test the network
    printf("XOR learned:\n");
    for (int i = 0; i < 4; i++) {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];

        double h_out[2];
        for (int j = 0; j < 2; j++) {
            double h_in = x1 * w_input_hidden[0][j] + x2 * w_input_hidden[1][j] + b_hidden[j];
            h_out[j] = sigmoid(h_in);
        }

        double z = h_out[0] * w_hidden_output[0] + h_out[1] * w_hidden_output[1] + b_output;
        double out = sigmoid(z);

        printf("Input: %.0f %.0f -> Output: %.4f (rounded: %d)\n",
               x1, x2, out, out >= 0.5 ? 1 : 0);
    }

    return 0;
}
