`timescale 1ns/1ps

module mlp_accelerator_tb;

  parameter IN_WIDTH = 32;
  parameter POS_DIM = 63;
  parameter DIR_DIM = 27;
  parameter OUT_DIM = 4;
  parameter SAMPLE_CNT = 1;
  parameter L1_UNITS = 256;
  parameter L2_UNITS = 256;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, ready, done;
  logic load_mode, load_valid, in_valid, out_valid;
  logic [15:0] load_addr;
  logic [IN_WIDTH-1:0] load_data;
  logic [IN_WIDTH-1:0] pos_in [POS_DIM];
  logic [IN_WIDTH-1:0] dir_in [DIR_DIM];
  logic [IN_WIDTH-1:0] out_rgb_sigma [OUT_DIM];
  logic out_ready, in_ready;

  mlp_accelerator dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .ready(ready), .done(done),
    .load_mode(load_mode), .load_addr(load_addr), .load_data(load_data), .load_valid(load_valid),
    .pos_in(pos_in), .dir_in(dir_in),
    .in_valid(in_valid), .in_ready(in_ready),
    .out_rgb_sigma(out_rgb_sigma), .out_valid(out_valid), .out_ready(out_ready)
  );

  task initialize_inputs;
    for (int i = 0; i < POS_DIM; i++) pos_in[i] = 32'h3F800000; // 1.0
    for (int j = 0; j < DIR_DIM; j++) dir_in[j] = 32'h40000000; // 2.0
  endtask

  task load_weights;
    input logic [3:0] sel;
    input int u_max;
    input int v_max;
    begin
      load_mode = 1;
      for (int u = 0; u < u_max; u++) begin
        for (int v = 0; v < v_max; v++) begin
          load_addr = {sel, u[7:0], v[7:0]};
          load_data = 32'h3F800000; // 1.0
          load_valid = 1;
          @(posedge clk);
        end
      end
      load_valid = 0;
      load_mode = 0;
    end
  endtask

  initial begin
    @(posedge clk);
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    $display("Loading layer1_weights...");
    load_weights(4'h0, L1_UNITS, POS_DIM);
    $display("Loading layer1_bias...");
    load_weights(4'h1, L1_UNITS, 1);
    $display("Loading layer2_weights...");
    load_weights(4'h2, L2_UNITS, L1_UNITS);
    $display("Loading layer2_bias...");
    load_weights(4'h3, L2_UNITS, 1);
    $display("Loading proj_weights...");
    load_weights(4'h4, OUT_DIM, L2_UNITS + DIR_DIM);
    $display("Loading proj_bias...");
    load_weights(4'h5, OUT_DIM, 1);

    @(posedge clk);
    initialize_inputs();

    in_valid = 1;
    start = 1;
    out_ready = 1;

    @(posedge clk);
    in_valid = 0;
    start = 0;

    wait(out_valid == 1);
    $display("Output:");
    for (int i = 0; i < OUT_DIM; i++) begin
      $display("out_rgb_sigma[%0d] = %h", i, out_rgb_sigma[i]);
    end

    $finish;
  end
endmodule

