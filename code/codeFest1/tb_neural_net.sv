`timescale 1ns / 1ps

module tb_spiking_neural_network;

    parameter WIDTH = 16;
    parameter N_INPUT = 4;
    parameter N_OUTPUT = 3;

    // DUT Signals
    logic clk, rst, load_params;
    logic signed [WIDTH-1:0] I_in [N_INPUT];
    logic signed [WIDTH-1:0] synapses [N_INPUT][N_OUTPUT];
    logic [WIDTH-1:0] new_V_threshold, new_leak_factor, new_refr_period;
    logic signed [WIDTH-1:0] new_V_max, new_V_min;
    logic spike_L2 [N_OUTPUT];

    // Clock generation
    always #5 clk = ~clk;

    // Instantiate DUT
    spiking_neural_network #(
        .WIDTH(WIDTH),
        .N_INPUT(N_INPUT),
        .N_OUTPUT(N_OUTPUT)
    ) dut (
        .clk(clk),
        .rst(rst),
        .I_in(I_in),
        .load_params(load_params),
        .new_V_threshold(new_V_threshold),
        .new_leak_factor(new_leak_factor),
        .new_refr_period(new_refr_period),
        .new_V_max(new_V_max),
        .new_V_min(new_V_min),
        .synapses(synapses),
        .spike_L2(spike_L2)
    );

    // Tasks
    task reset_dut;
        begin
            rst = 1;
            #20;
            rst = 0;
        end
    endtask

    task load_neuron_params(
        input [WIDTH-1:0] threshold, leak, refract,
        input signed [WIDTH-1:0] v_max, v_min
    );
        begin
            new_V_threshold = threshold;
            new_leak_factor = leak;
            new_refr_period = refract;
            new_V_max = v_max;
            new_V_min = v_min;
            load_params = 1;
            #10;
            load_params = 0;
        end
    endtask

    // Simulation
    initial begin
        $display("=== Starting Spiking Neural Network Testbench ===");
        $dumpfile("network.vcd");     // For GTKWave
        $dumpvars(0, tb_spiking_neural_network);

        clk = 0;
        rst = 0;
        load_params = 0;

        // Reset DUT
        reset_dut();

        // Load initial neuron parameters
        load_neuron_params(
            1000,   // threshold
            50,     // leak
            5,      // refractory
            32767,  // Vmax
            -32768  // Vmin
        );

        // ========== Test 1: Excitatory Burst ==========
        $display("\n[Test 1] Excitatory spike burst from L1[0]");

        for (int i = 0; i < N_INPUT; i++) begin
            for (int j = 0; j < N_OUTPUT; j++) begin
                synapses[i][j] = 500; // All excitatory
            end
        end

        I_in[0] = 2000;  // Only L1[0] fires
        I_in[1] = 0;
        I_in[2] = 0;
        I_in[3] = 0;

        repeat (10) @(posedge clk);
        for (int i = 0; i < N_INPUT; i++) I_in[i] = 0;

        repeat (5) @(posedge clk);

        // ========== Test 2: Inhibitory Synapse ==========
        $display("\n[Test 2] Inhibitory weight from L1[1] to L2[0]");

        synapses[1][0] = -1000;  // Inhibitory input to L2[0]
        I_in[1] = 3000;  // Should suppress L2[0]
        repeat (5) @(posedge clk);
        I_in[1] = 0;

        repeat (5) @(posedge clk);

        // ========== Test 3: Saturation Test ==========
        $display("\n[Test 3] Saturation test with huge weight");

        synapses[2][1] = 30000; // High weight to test V_mem saturation
        I_in[2] = 3000;
        repeat (3) @(posedge clk);
        I_in[2] = 0;

        repeat (5) @(posedge clk);

        // ========== Test 4: Mid-run Parameter Change ==========
        $display("\n[Test 4] Change parameters mid-simulation");

        load_neuron_params(
            2000, // higher threshold
            100,  // more leak
            3,    // shorter refractory
            20000, -20000
        );
        repeat (3) @(posedge clk);

        // ========== Test 5: Randomized Inputs ==========
        $display("\n[Test 5] Random inputs to all L1 neurons");
        for (int i = 0; i < 5; i++) begin
            for (int j = 0; j < N_INPUT; j++) begin
                I_in[j] = $urandom_range(-2000, 3000);
            end
            @(posedge clk);
            $display("[%0t ns] I_in: %p | L2 Spikes: %p", $time, I_in, spike_L2);
        end

        $display("=== Testbench Complete ===");
        $finish;
    end

endmodule
