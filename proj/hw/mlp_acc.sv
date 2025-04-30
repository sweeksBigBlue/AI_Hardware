module mlp_accelerator #(
    parameter integer IN_WIDTH   = 32,         // Float width
    parameter integer POS_DIM    = 63,         // Encoded position dims
    parameter integer DIR_DIM    = 27,         // Encoded direction dims
    parameter integer OUT_DIM    = 4,          // RGB + sigma
    parameter integer SAMPLE_CNT = 65536,      // Total number of samples
    parameter integer L1_UNITS   = 256,        // Neurons in first MLP layer
    parameter integer L2_UNITS   = 256         // Neurons in second MLP layer
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  start,
    output logic                  ready,
    output logic                  done,

    input  logic                  load_mode,         // Set high to load weights
    input  logic [15:0]           load_addr,         // Address for loading weights
    input  logic [IN_WIDTH-1:0]   load_data,         // Data for loading
    input  logic                  load_valid,        // Valid signal for loading

    input  wire [IN_WIDTH-1:0]   pos_in [POS_DIM],
    input  wire [IN_WIDTH-1:0]   dir_in [DIR_DIM],
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [IN_WIDTH-1:0]   out_rgb_sigma [OUT_DIM],
    output logic                  out_valid,
    input  logic                  out_ready
);

  typedef enum logic [2:0] {
    IDLE, READ, COMPUTE_L1, COMPUTE_L2, COMPUTE_PROJ, WRITE, DONE
  } state_t;

  state_t state, next_state;
  logic [$clog2(SAMPLE_CNT)-1:0] sample_idx;

  logic [IN_WIDTH-1:0] pos_reg [POS_DIM];
  logic [IN_WIDTH-1:0] dir_reg [DIR_DIM];
  logic [IN_WIDTH-1:0] out_reg [OUT_DIM];

  logic signed [IN_WIDTH-1:0] layer1_out [L1_UNITS];
  logic signed [IN_WIDTH-1:0] layer1_weights [L1_UNITS][POS_DIM];
  logic signed [IN_WIDTH-1:0] layer1_bias [L1_UNITS];

  logic signed [IN_WIDTH-1:0] layer2_out [L2_UNITS];
  logic signed [IN_WIDTH-1:0] layer2_weights [L2_UNITS][L1_UNITS];
  logic signed [IN_WIDTH-1:0] layer2_bias [L2_UNITS];

  logic signed [IN_WIDTH-1:0] proj_weights [OUT_DIM][L2_UNITS + DIR_DIM];
  logic signed [IN_WIDTH-1:0] proj_bias [OUT_DIM];

  logic signed [IN_WIDTH-1:0] acc;

  function logic signed [IN_WIDTH-1:0] relu(input logic signed [IN_WIDTH-1:0] x);
    return (x < 0) ? 0 : x;
  endfunction

  function logic signed [IN_WIDTH-1:0] sigmoid(input logic signed [IN_WIDTH-1:0] x);
    return x / (1 + ((x < 0) ? -x : x)); // Approximation
  endfunction

  function logic signed [IN_WIDTH-1:0] softplus(input logic signed [IN_WIDTH-1:0] x);
    return (x < 0) ? 0 : x; // Approximation
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= IDLE;
    else
      state <= next_state;
  end

  always_comb begin
    next_state = state;
    case (state)
      IDLE: if (start) next_state = READ;
      READ: if (in_valid) next_state = COMPUTE_L1;
      COMPUTE_L1: next_state = COMPUTE_L2;
      COMPUTE_L2: next_state = COMPUTE_PROJ;
      COMPUTE_PROJ: next_state = WRITE;
      WRITE: next_state = (sample_idx == SAMPLE_CNT-1) ? DONE : READ;
      DONE: next_state = IDLE;
    endcase
  end

  assign in_ready = (state == READ);
  assign out_valid = (state == WRITE);
  assign ready = (state == IDLE);
  assign done = (state == DONE);

  always_ff @(posedge clk) begin
    if (state == READ && in_valid) begin
      pos_reg <= pos_in;
      dir_reg <= dir_in;
    end else begin
      pos_reg <= pos_reg;
      dir_reg <= dir_reg;
    end
  end

  always_ff @(posedge clk) begin
    if (state == COMPUTE_L1) begin
      for (int u = 0; u < L1_UNITS; u++) begin
        acc <= layer1_bias[u];
        for (int i = 0; i < POS_DIM; i++) begin
          acc += (layer1_weights[u][i] * pos_reg[i]);
        end
        layer1_out[u] <= relu(acc);
      end
    end else begin
      layer1_out <= layer1_out;
    end
  end

  always_ff @(posedge clk) begin
    if (state == COMPUTE_L2) begin
      for (int u = 0; u < L2_UNITS; u++) begin
        acc <= layer2_bias[u];
        for (int i = 0; i < L1_UNITS; i++) begin
          acc += (layer2_weights[u][i] * layer1_out[i]);
        end
        layer2_out[u] <= relu(acc);
      end
    end else begin
      layer2_out <= layer2_out;
    end
  end

  always_ff @(posedge clk) begin
    if (state == COMPUTE_PROJ) begin
      for (int o = 0; o < OUT_DIM; o++) begin
        acc <= proj_bias[o];
        for (int i = 0; i < L2_UNITS; i++) acc += proj_weights[o][i] * layer2_out[i];
        for (int j = 0; j < DIR_DIM; j++) acc += proj_weights[o][L2_UNITS + j] * dir_reg[j];

        if (o < 3)
          out_reg[o] <= sigmoid(acc); // RGB
        else
          out_reg[o] <= softplus(acc); // Sigma
      end
    end else begin
      out_reg <= out_reg;
    end
  end

  always_ff @(posedge clk) begin
    if (state == WRITE && out_ready) begin
      out_rgb_sigma <= out_reg;
    end else begin
      out_rgb_sigma <= out_rgb_sigma;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sample_idx <= 0;
    else if (state == WRITE && out_ready) sample_idx <= sample_idx + 1;
    else if (state == IDLE) sample_idx <= 0;
    else sample_idx <= sample_idx;
  end

  always_ff @(posedge clk) begin
    if (load_mode && load_valid) begin
      case (load_addr[15:12])
        4'h0: layer1_weights[load_addr[11:8]][load_addr[7:0]] <= load_data;
        4'h1: layer1_bias[load_addr[7:0]] <= load_data;
        4'h2: layer2_weights[load_addr[11:8]][load_addr[7:0]] <= load_data;
        4'h3: layer2_bias[load_addr[7:0]] <= load_data;
        4'h4: proj_weights[load_addr[11:8]][load_addr[7:0]] <= load_data;
        4'h5: proj_bias[load_addr[7:0]] <= load_data;
        default: ;
      endcase
    end else begin
      // No weight update
    end
  end

endmodule

