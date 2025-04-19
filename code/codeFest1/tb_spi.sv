`timescale 1ns/1ps

module tb_spi_interface;

    // Parameters
    parameter int WIDTH = 16;
    parameter int N_INPUT = 4;
    parameter int N_OUTPUT = 3;
    parameter int PARAM_BYTES = 10;
    parameter int SYNAPSE_BYTES = 3;
    parameter int SPIKE_READ_BITS = N_OUTPUT;

    // SPI Interface signals
    logic clk, reset;
    logic sclk, cs, mosi;
    logic miso;

    // Outputs from SPI interface
    logic [WIDTH-1:0] param_threshold, param_leak, param_refr;
    logic signed [WIDTH-1:0] param_vmax, param_vmin;
    logic [$clog2(N_INPUT)-1:0] syn_src;
    logic [$clog2(N_OUTPUT)-1:0] syn_dst;
    logic signed [WIDTH-1:0] syn_weight;
    logic load_params, update_synapse, net_reset;
    logic spike_out [N_OUTPUT];

    // Instantiate SPI Interface
    spi_interface #(
        .WIDTH(WIDTH),
        .N_INPUT(N_INPUT),
        .N_OUTPUT(N_OUTPUT),
        .PARAM_BYTES(PARAM_BYTES),
        .SYNAPSE_BYTES(SYNAPSE_BYTES),
        .SPIKE_READ_BITS(SPIKE_READ_BITS)
    ) dut (
        .clk(clk), .reset(reset),
        .sclk(sclk), .cs(cs), .mosi(mosi), .miso(miso),
        .param_threshold(param_threshold),
        .param_leak(param_leak),
        .param_refr(param_refr),
        .param_vmax(param_vmax),
        .param_vmin(param_vmin),
        .syn_src(syn_src),
        .syn_dst(syn_dst),
        .syn_weight(syn_weight),
        .load_params(load_params),
        .update_synapse(update_synapse),
        .net_reset(net_reset),
        .spike_out(spike_out)
    );

    // Clock
    always #5 clk = ~clk;

    // Test SPI sending function
    task spi_send_byte(input [7:0] byte);
        for (int i = 7; i >= 0; i--) begin
            mosi = byte[i];
            #5 sclk = 1;
            #5 sclk = 0;
        end
    endtask

    // Stimulus
    initial begin
        $display("=== Starting SPI Interface Testbench ===");
        clk = 0; reset = 0;
        sclk = 0; cs = 1; mosi = 0;

        // Initialize spike output pattern
        spike_out[0] = 1;
        spike_out[1] = 0;
        spike_out[2] = 1;

        #20 reset = 1; #20 reset = 0;

        // === Test 1: Load Neuron Parameters ===
        $display("-> Sending Neuron Parameters");
        cs = 0;
        spi_send_byte(8'h01); // Command
        spi_send_byte(8'h03); spi_send_byte(8'hE8); // threshold = 1000
        spi_send_byte(8'h00); spi_send_byte(8'h64); // leak = 100
        spi_send_byte(8'h00); spi_send_byte(8'h05); // refr = 5
        spi_send_byte(8'h7F); spi_send_byte(8'hFF); // vmax = 32767
        spi_send_byte(8'h80); spi_send_byte(8'h00); // vmin = -32768
        cs = 1; #20;

        // Check param outputs
        $display("Threshold: %0d", param_threshold);
        $display("Leak     : %0d", param_leak);
        $display("Refr     : %0d", param_refr);
        $display("Vmax     : %0d", param_vmax);
        $display("Vmin     : %0d", param_vmin);

        // === Test 2: Load Synapse ===
        $display("-> Sending Synapse Data");
        cs = 0;
        spi_send_byte(8'h02); // Command
        spi_send_byte(8'h01); // src = 1
        spi_send_byte(8'h02); // dst = 2
        spi_send_byte(8'hF6); // weight = -10 (signed 8-bit)
        cs = 1; #20;

        $display("Syn Src : %0d", syn_src);
        $display("Syn Dst : %0d", syn_dst);
        $display("Syn Wt  : %0d", syn_weight);

        // === Test 3: Reset Command ===
        $display("-> Sending Reset Command");
        cs = 0;
        spi_send_byte(8'h03); // Command
        cs = 1; #20;
        $display("Net Reset: %b", net_reset);

        // === Test 4: Read Spikes ===
        $display("-> Reading Spikes from Network");
        cs = 0;
        spi_send_byte(8'h04); // Command
        #10; // allow shift
        for (int i = 0; i < N_OUTPUT; i++) begin
            #5 sclk = 1;
            #5 sclk = 0;
            $display("MISO [%0d] = %b", i, miso);
        end
        cs = 1;

        $display("=== SPI Interface Test Completed ===");
        $finish;
    end
endmodule
