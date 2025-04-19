// Q-Table Hardware: SRAM-Based Implementation for 25 states × 4 actions
// Fixed-point 16-bit Q-values, dual-port SRAM for simultaneous read/write

module QTable #(parameter STATE_BITS = 5, // log2(25) = 5
                parameter ACTION_BITS = 2, // log2(4) = 2
                parameter Q_WIDTH = 16)(
    input logic clk,
    input logic rst,

    // Read interface
    input logic [STATE_BITS-1:0] read_state,
    input logic [ACTION_BITS-1:0] read_action,
    output logic [Q_WIDTH-1:0] q_read_data,

    // Write interface
    input logic we, // write enable
    input logic [STATE_BITS-1:0] write_state,
    input logic [ACTION_BITS-1:0] write_action,
    input logic [Q_WIDTH-1:0] q_write_data
);

    // 25 states x 4 actions
    localparam ENTRY_COUNT = 25 * 4;
    logic [Q_WIDTH-1:0] q_table [0:ENTRY_COUNT-1];

    logic [6:0] read_addr, write_addr;
    assign read_addr  = {read_state, read_action};
    assign write_addr = {write_state, write_action};

    // Read
    assign q_read_data = q_table[read_addr];

    // Write
    always_ff @(posedge clk) begin
        if (we) begin
            q_table[write_addr] <= q_write_data;
        end
    end

endmodule
