module mlp_accelerator #(
    parameter integer IN_WIDTH   = 32,         // Float width
    parameter integer POS_DIM    = 63,         // Encoded position dims
    parameter integer DIR_DIM    = 27,         // Encoded direction dims
    parameter integer OUT_DIM    = 4,          // RGB + sigma
    parameter integer SAMPLE_CNT = 65536       // Total number of samples
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  start,
    output logic                  ready,
    output logic                  done,

    input  logic [IN_WIDTH-1:0]   pos_in [POS_DIM],
    input  logic [IN_WIDTH-1:0]   dir_in [DIR_DIM],
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [IN_WIDTH-1:0]   out_rgb_sigma [OUT_DIM],
    output logic                  out_valid,
    input  logic                  out_ready
);

  typedef enum logic [2:0] {
    IDLE, READ, COMPUTE, WRITE, DONE
  } state_t;

  state_t state, next_state;
  logic [$clog2(SAMPLE_CNT)-1:0] sample_idx;
  logic compute_enable;

  // Dummy internal registers for demo
  logic [IN_WIDTH-1:0] pos_reg [POS_DIM];
  logic [IN_WIDTH-1:0] dir_reg [DIR_DIM];
  logic [IN_WIDTH-1:0] out_reg [OUT_DIM];

  // State transitions
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= IDLE;
    else
      state <= next_state;
  end

  // Next state logic
  always_comb begin
    next_state = state;
    case (state)
      IDLE: if (start) next_state = READ;
      READ: if (in_valid) next_state = COMPUTE;
      COMPUTE: next_state = WRITE;
      WRITE: if (out_ready) next_state = (sample_idx == SAMPLE_CNT-1) ? DONE : READ;
      DONE: next_state = IDLE;
    endcase
  end

  // Handshake controls
  assign in_ready = (state == READ);
  assign out_valid = (state == WRITE);
  assign ready = (state == IDLE);
  assign done = (state == DONE);
  assign compute_enable = (state == COMPUTE);

  // Register input data
  always_ff @(posedge clk) begin
    if (state == READ && in_valid) begin
      pos_reg <= pos_in;
      dir_reg <= dir_in;
    end else begin
      pos_reg <= pos_reg;
      dir_reg <= dir_reg;
    end
  end

  // Simulated MLP processing (mock for now)
  always_ff @(posedge clk) begin
    if (compute_enable) begin
      for (int k = 0; k < OUT_DIM; k++)
        out_reg[k] <= $shortrealtobits(1.0); // Placeholder constant output
    end else begin
      out_reg <= out_reg;
    end
  end

  // Output assignments
  always_ff @(posedge clk) begin
    if (state == WRITE && out_ready) begin
      out_rgb_sigma <= out_reg;
    end else begin
      out_rgb_sigma <= out_rgb_sigma;
    end
  end

  // Sample counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sample_idx <= 0;
    else if (state == WRITE && out_ready) sample_idx <= sample_idx + 1;
    else if (state == IDLE) sample_idx <= 0;
    else sample_idx <= sample_idx;
  end

endmodule

