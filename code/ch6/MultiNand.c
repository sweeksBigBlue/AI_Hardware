#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

// Sigmoid and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// NAND target function
int nand_target(int *inputs, int size) {
    int and_result = 1;
    for (int i = 0; i < size; i++) {
        and_result &= inputs[i];
    }
    return !and_result;
}

// Generate binary input combinations
void generate_inputs(int **data, int n_inputs, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_inputs; j++) {
            data[i][j] = (i >> (n_inputs - j - 1)) & 1;
        }
    }
}

// Forward pass
double forward(int *inputs, double *weights, double bias, int size) {
    double z = 0.0;
    for (int i = 0; i < size; i++) {
        z += inputs[i] * weights[i];
    }
    z += bias;
    return sigmoid(z);
}

// Training function
void train(double *weights, double *bias, int n_inputs, double lr, int epochs) {
    int n_samples = 1 << n_inputs;

    int **train_data = malloc(n_samples * sizeof(int *));
    for (int i = 0; i < n_samples; i++) {
        train_data[i] = malloc(n_inputs * sizeof(int));
    }

    generate_inputs(train_data, n_inputs, n_samples);

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n_samples; i++) {
            int *x = train_data[i];
            int y = nand_target(x, n_inputs);

            double z = 0.0;
            for (int j = 0; j < n_inputs; j++) {
                z += weights[j] * x[j];
            }
            z += *bias;

            double output = sigmoid(z);
            double error = y - output;
            double delta = error * sigmoid_derivative(z);

            for (int j = 0; j < n_inputs; j++) {
                weights[j] += lr * delta * x[j];
            }
            *bias += lr * delta;
        }
    }

    for (int i = 0; i < n_samples; i++) free(train_data[i]);
    free(train_data);
}

// Help menu
void print_help(const char *prog_name) {
    printf("Usage: %s -n num_inputs\n", prog_name);
    printf("\nOptions:\n");
    printf("  -n  Number of binary inputs (e.g. -n 3)\n");
    printf("  -h  Show this help message\n\n");
    printf("Example:\n");
    printf("  %s -n 3\n", prog_name);
}

int main(int argc, char *argv[]) {
    int n_inputs = 0;
    int opt;

    while ((opt = getopt(argc, argv, "n:h")) != -1) {
        switch (opt) {
            case 'n':
                n_inputs = atoi(optarg);
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (n_inputs <= 0) {
        fprintf(stderr, "Error: You must specify -n with a positive number of inputs.\n\n");
        print_help(argv[0]);
        return 1;
    }

    int n_samples = 1 << n_inputs;
    double lr = 0.1;
    int epochs = 10000;
    double *weights = calloc(n_inputs, sizeof(double));
    double bias = 0.0;

    train(weights, &bias, n_inputs, lr, epochs);

    printf("Trained weights and bias:\n");
    for (int i = 0; i < n_inputs; i++) {
        printf("w[%d] = %.4f\n", i, weights[i]);
    }
    printf("bias = %.4f\n\n", bias);

    printf("Testing NAND gate with %d inputs:\n", n_inputs);
    for (int i = 0; i < n_samples; i++) {
        int inputs[n_inputs];
        for (int j = 0; j < n_inputs; j++) {
            inputs[j] = (i >> (n_inputs - j - 1)) & 1;
        }

        double out = forward(inputs, weights, bias, n_inputs);
        int predicted = out >= 0.5 ? 1 : 0;

        printf("Input: ");
        for (int j = 0; j < n_inputs; j++) {
            printf("%d ", inputs[j]);
        }
        printf("-> Output: %.4f (rounded: %d)\n", out, predicted);
    }

    free(weights);
    return 0;
}
