module spiking_neural_network #(
    parameter int WIDTH = 16,      // Bit width for calculations
    parameter int N_INPUT = 4,     // Number of input neurons
    parameter int N_OUTPUT = 3     // Number of output neurons
) (
    input  logic clk, rst,      
    input  logic signed [WIDTH-1:0] I_in [N_INPUT], // External input currents
    input  logic load_params,  // Load new neuron and synapse parameters
    input  logic [WIDTH-1:0] new_V_threshold, new_leak_factor, new_refr_period,
    input  logic signed [WIDTH-1:0] new_V_max, new_V_min,
    input  logic signed [WIDTH-1:0] synapses [N_INPUT][N_OUTPUT], // Programmable synaptic weights
    output logic spike_L2 [N_OUTPUT]  // Final network spike outputs
);

    logic spike_L1 [N_INPUT];  // Spikes from Layer 1 neurons
    logic signed [WIDTH+4:0] I_syn [N_OUTPUT]; // Widened for safe accumulation

    // Internal register for synaptic weights
    logic signed [WIDTH-1:0] synapse_matrix [N_INPUT][N_OUTPUT];

    // Load synaptic weights when load_params is high
    always_ff @(posedge clk) begin
        if (load_params) begin
            for (int i = 0; i < N_INPUT; i++) begin
                for (int j = 0; j < N_OUTPUT; j++) begin
                    synapse_matrix[i][j] <= synapses[i][j];
                end
            end
        end
    end

    // Instantiate Layer 1 neurons
    genvar i;
    generate
        for (i = 0; i < N_INPUT; i++) begin : L1_NEURONS
            lif_neuron #(.WIDTH(WIDTH)) neuron_L1 (
                .clk(clk),
                .rst(rst),
                .I_in(I_in[i]),
                .load_params(load_params),
                .new_V_threshold(new_V_threshold),
                .new_leak_factor(new_leak_factor),
                .new_refr_period(new_refr_period),
                .new_V_max(new_V_max),
                .new_V_min(new_V_min),
                .spike(spike_L1[i])
            );
        end
    endgenerate

    // Compute synaptic currents to Layer 2
    integer j, k;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < N_OUTPUT; j = j + 1) begin
                I_syn[j] <= 0;
            end
        end else begin
            for (j = 0; j < N_OUTPUT; j = j + 1) begin
                I_syn[j] <= 0; // Start from zero each cycle
                for (k = 0; k < N_INPUT; k = k + 1) begin
                    if (spike_L1[k]) begin
                        I_syn[j] <= I_syn[j] + signed'(synapse_matrix[k][j]);
                    end
                end
            end
        end
    end

    // Instantiate Layer 2 neurons
    generate
        for (j = 0; j < N_OUTPUT; j++) begin : L2_NEURONS
            lif_neuron #(.WIDTH(WIDTH)) neuron_L2 (
                .clk(clk),
                .rst(rst),
                .I_in(I_syn[j][WIDTH-1:0]), // Truncate wider sum safely
                .load_params(load_params),
                .new_V_threshold(new_V_threshold),
                .new_leak_factor(new_leak_factor),
                .new_refr_period(new_refr_period),
                .new_V_max(new_V_max),
                .new_V_min(new_V_min),
                .spike(spike_L2[j])
            );
        end
    endgenerate

endmodule
