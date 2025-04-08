#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void print_help(const char *prog_name) {
  printf("Usage: %s -i val1,val2 -w val1,val2 -b bias\n", prog_name);
  printf("\nOptions:\n");
  printf("  -i  Two comma-separated input values (e.g. -i 0.6,0.9)\n");
  printf("  -w  Two comma-separated weight values (e.g. -w 1.0,-1.5)\n");
  printf("  -b  A single bias value (e.g. -b 0.2)\n");
  printf("  -h  Show this help message\n\n");
  printf("Example:\n");
  printf("  %s -i 0.6,0.9 -w 1.0,-1.5 -b 0.2\n", prog_name);
}

// Sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Perceptron computation
double perceptron(double inputs[], double weights[], double bias, int size) {
  double z = 0.0;
  for (int i = 0; i < size; i++) {
    z += inputs[i] * weights[i];
  }
  z += bias;
  return sigmoid(z);
}

// Argument parser
int parse_arguments(int argc, char *argv[], double inputs[], double weights[],
                    double *bias) {
  int got_input = 0, got_weights = 0, got_bias = 0;
  int opt;

  while ((opt = getopt(argc, argv, "i:w:b:h")) != -1) {
    switch (opt) {
    case 'i':
      if (sscanf(optarg, "%lf,%lf", &inputs[0], &inputs[1]) != 2) {
        fprintf(stderr, "Invalid input format. Use -i val1,val2\n");
        return 1;
      }
      got_input = 1;
      break;
    case 'w':
      if (sscanf(optarg, "%lf,%lf", &weights[0], &weights[1]) != 2) {
        fprintf(stderr, "Invalid weight format. Use -w val1,val2\n");
        return 1;
      }
      got_weights = 1;
      break;
    case 'b':
      *bias = atof(optarg);
      got_bias = 1;
      break;
    case 'h':
      print_help(argv[0]);
      exit(0);
    default:
      print_help(argv[0]);
      return 1;
    }
  }

  if (!got_input || !got_weights || !got_bias) {
    fprintf(stderr, "Error: All options -i, -w, and -b are required.\n\n");
    print_help(argv[0]);
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  double inputs[2] = {0.0, 0.0};
  double weights[2] = {0.0, 0.0};
  double bias = 0.0;

  if (parse_arguments(argc, argv, inputs, weights, &bias) != 0) {
    return 1;
  }

  double output = perceptron(inputs, weights, bias, 2);
  printf("Perceptron output: %.4f\n", output);

  return 0;
}
