// State Tracker FSM for 5x5 Grid (FrozenLake environment)
// Handles (x, y) position, obstacle detection, and reset logic

module StateFSM #(parameter X_BITS = 3,  // log2(5) = 3
                   parameter Y_BITS = 3)(
    input logic clk,
    input logic rst,
    input logic [1:0] action,  // 2-bit input: 00=up, 01=down, 10=left, 11=right
    input logic step,         // step trigger (advance 1 time step)

    output logic [X_BITS-1:0] pos_x,
    output logic [Y_BITS-1:0] pos_y,
    output logic is_hole,
    output logic is_goal
);

    // Internal position registers
    logic [X_BITS-1:0] x;
    logic [Y_BITS-1:0] y;

    // Hole ROM: hardcoded 4 positions
    function logic is_hole_pos(input logic [2:0] x_in, input logic [2:0] y_in);
        return (x_in == 1 && y_in == 0) ||
               (x_in == 3 && y_in == 1) ||
               (x_in == 4 && y_in == 2) ||
               (x_in == 1 && y_in == 3);
    endfunction

    function logic is_goal_pos(input logic [2:0] x_in, input logic [2:0] y_in);
        return (x_in == 4 && y_in == 4);
    endfunction

    // Position update
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            x <= 0;
            y <= 0;
        end else if (step) begin
            case (action)
                2'b00: begin
                    if (y > 0) y <= y - 1;
                    else y <= y;
                end
                2'b01: begin
                    if (y < 4) y <= y + 1;
                    else y <= y;
                end
                2'b10: begin
                    if (x > 0) x <= x - 1;
                    else x <= x;
                end
                2'b11: begin
                    if (x < 4) x <= x + 1;
                    else x <= x;
                end
                default: begin
                    x <= x;
                    y <= y;
                end
            endcase
        end
    end

    assign pos_x = x;
    assign pos_y = y;
    assign is_hole = is_hole_pos(x, y);
    assign is_goal = is_goal_pos(x, y);

endmodule
