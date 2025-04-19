module spi_interface #(
    parameter int WIDTH = 16,
    parameter int N_INPUT = 4,
    parameter int N_OUTPUT = 3,
    parameter int PARAM_BYTES = 10,     // 5 fields × 2 bytes
    parameter int SYNAPSE_BYTES = 3,    // src, dst, weight
    parameter int SPIKE_READ_BITS = N_OUTPUT
)(
    input  logic clk, reset,
    input  logic sclk, cs, mosi,
    output logic miso,

    // Configuration outputs
    output logic [WIDTH-1:0] param_threshold,
    output logic [WIDTH-1:0] param_leak,
    output logic [WIDTH-1:0] param_refr,
    output logic signed [WIDTH-1:0] param_vmax,
    output logic signed [WIDTH-1:0] param_vmin,
    output logic [$clog2(N_INPUT)-1:0] syn_src,
    output logic [$clog2(N_OUTPUT)-1:0] syn_dst,
    output logic signed [WIDTH-1:0] syn_weight,
    output logic load_params,
    output logic update_synapse,
    output logic net_reset,

    // Spike read-back
    input  logic spike_out [N_OUTPUT]
);

    typedef enum logic [2:0] {
        IDLE, CMD, LOAD_PARAMS, LOAD_SYNAPSE, READ_SPIKE
    } spi_state_t;

    spi_state_t state;

    logic [2:0] bit_cnt;              // Bits received for current byte (0–7)
    logic [$clog2(PARAM_BYTES):0] byte_cnt; // Which byte in the packet
    logic [7:0] shift_reg;           // Assembles one byte
    logic [7:0] byte_buffer [0:9];   // Holds up to 10 bytes (PARAM_BYTES)

    logic [$clog2(SPIKE_READ_BITS)-1:0] spike_read_index;

    always_ff @(posedge sclk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            bit_cnt <= 0;
            byte_cnt <= 0;
            shift_reg <= 0;
            miso <= 0;
            load_params <= 0;
            update_synapse <= 0;
            net_reset <= 0;
        end else if (!cs) begin
            shift_reg <= {shift_reg[6:0], mosi};
            bit_cnt <= bit_cnt + 1;

            if (bit_cnt == 7) begin
                // Full byte assembled
                case (state)
                    IDLE: begin
                        case ({shift_reg[6:0], mosi})
                            8'h01: begin state <= LOAD_PARAMS; byte_cnt <= 0; end
                            8'h02: begin state <= LOAD_SYNAPSE; byte_cnt <= 0; end
                            8'h03: begin net_reset <= 1; state <= IDLE; end
                            8'h04: begin state <= READ_SPIKE; spike_read_index <= 0; end
                            default: state <= IDLE;
                        endcase
                    end

                    LOAD_PARAMS: begin
                        if (byte_cnt < PARAM_BYTES) begin
                            byte_buffer[byte_cnt] <= {shift_reg[6:0], mosi};
                            byte_cnt <= byte_cnt + 1;
                        end
                        if (byte_cnt == PARAM_BYTES - 1) begin
                            // Assume each parameter is 2 bytes: big-endian
                            param_threshold <= {byte_buffer[0], byte_buffer[1]};
                            param_leak      <= {byte_buffer[2], byte_buffer[3]};
                            param_refr      <= {byte_buffer[4], byte_buffer[5]};
                            param_vmax      <= {byte_buffer[6], byte_buffer[7]};
                            param_vmin      <= {byte_buffer[8], byte_buffer[9]};
                            load_params <= 1;
                            state <= IDLE;
                        end
                    end

                    LOAD_SYNAPSE: begin
                        if (byte_cnt < SYNAPSE_BYTES) begin
                            byte_buffer[byte_cnt] <= {shift_reg[6:0], mosi};
                            byte_cnt <= byte_cnt + 1;
                        end
                        if (byte_cnt == SYNAPSE_BYTES - 1) begin
                            syn_src    <= byte_buffer[0];
                            syn_dst    <= byte_buffer[1];
                            syn_weight <= {{(WIDTH-8){byte_buffer[2][7]}}, byte_buffer[2]}; // sign-extend
                            update_synapse <= 1;
                            state <= IDLE;
                        end
                    end

                    READ_SPIKE: begin
                        miso <= spike_out[spike_read_index];
                        spike_read_index <= spike_read_index + 1;
                        if (spike_read_index == SPIKE_READ_BITS - 1) begin
                            state <= IDLE;
                        end
                    end
                endcase
                bit_cnt <= 0;
            end
        end else begin
            // When CS goes high, reset state machine
            state <= IDLE;
            bit_cnt <= 0;
            byte_cnt <= 0;
            load_params <= 0;
            update_synapse <= 0;
            net_reset <= 0;
            miso <= 0;
        end
    end

endmodule
