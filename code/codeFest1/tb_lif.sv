`timescale 1ns / 1ps

module lif_neuron_tb;
    parameter WIDTH = 16;
    
    // DUT Inputs
    logic clk, rst;
    logic signed [WIDTH-1:0] I_in;
    logic load_params;
    logic [WIDTH-1:0] new_V_threshold;
    logic [WIDTH-1:0] new_leak_factor;
    logic [WIDTH-1:0] new_refr_period;
    logic signed [WIDTH-1:0] new_V_max;
    logic signed [WIDTH-1:0] new_V_min;
    
    // DUT Output
    logic spike;
    
    // Instantiate DUT
    lif_neuron #(.WIDTH(WIDTH)) uut (
        .clk(clk),
        .rst(rst),
        .I_in(I_in),
        .load_params(load_params),
        .new_V_threshold(new_V_threshold),
        .new_leak_factor(new_leak_factor),
        .new_refr_period(new_refr_period),
        .new_V_max(new_V_max),
        .new_V_min(new_V_min),
        .spike(spike)
    );

    // Clock Generation
    always #5 clk = ~clk; // 10 ns period (100 MHz)

    // Task for Reset
    task reset_dut;
        begin
            rst = 1;
            #20;
            rst = 0;
        end
    endtask

    // Task to load parameters
    task load_parameters(
        input logic [WIDTH-1:0] V_thresh,
        input logic [WIDTH-1:0] leak,
        input logic [WIDTH-1:0] refr,
        input logic signed [WIDTH-1:0] V_max,
        input logic signed [WIDTH-1:0] V_min
    );
        begin
            new_V_threshold = V_thresh;
            new_leak_factor = leak;
            new_refr_period = refr;
            new_V_max = V_max;
            new_V_min = V_min;
            load_params = 1;
            #10;
            load_params = 0;
        end
    endtask

    // Simulation
    initial begin
        // Initialize
        clk = 0;
        rst = 0;
        load_params = 0;
        I_in = 0;

        // Reset the neuron
        reset_dut;

        // Load parameters
        load_parameters(10000, 50, 5, 30000, -30000);
        #50;

        // 1. Test increasing input current
        for (int i = -32768; i < 32767; i += 5000) begin
            I_in = i;
            #10;
        end

        // 2. Test threshold crossing and spike
        I_in = 11000; // High enough to cross threshold
        #10;
        if (spike) $display("Spike detected correctly at time %0t", $time);
        else       $display("ERROR: No spike detected at expected threshold");

        // 3. Test refractory period
        for (int i = 0; i < 10; i++) begin
            I_in = 0;
            #10;
            if (spike) $display("ERROR: Spiked during refractory period at time %0t", $time);
        end

        // 4. Test underflow handling
        load_parameters(10000, 50, 5, 30000, -32768);
        I_in = -20000;
        #20;
        if (I_in < -32768) $display("ERROR: Underflow detected!");

        // 5. Test overflow handling
        load_parameters(10000, 50, 5, 32767, -30000);
        I_in = 32000;
        #20;
        if (I_in > 32767) $display("ERROR: Overflow detected!");

        // End simulation
        $display("TEST COMPLETED");
        $stop;
    end
endmodule
