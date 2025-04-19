module lif_neuron #(
    parameter int WIDTH = 16  // Bit width of potential and current
) (
    input  logic clk,          
    input  logic rst,          
    input  logic signed [WIDTH-1:0] I_in,  // Input current
    input  logic load_params,  // Signal to load new parameters
    input  logic [WIDTH-1:0] new_V_threshold, // New spike threshold
    input  logic [WIDTH-1:0] new_leak_factor, // New leakage per cycle
    input  logic [WIDTH-1:0] new_refr_period, // New refractory period
    input  logic signed [WIDTH-1:0] new_V_max, // New max potential
    input  logic signed [WIDTH-1:0] new_V_min, // New min potential
    output logic spike          
);

    // Internal registers for programmable parameters
    logic signed [WIDTH-1:0] V_threshold;
    logic signed [WIDTH-1:0] leak_factor;
    logic signed [WIDTH-1:0] V_max, V_min;
    logic [$clog2(WIDTH):0] refr_period; // Enough bits for refractory period
    logic [$clog2(WIDTH):0] refr_count;  // Refractory counter

    logic signed [WIDTH-1:0] V_mem;  // Membrane potential
    logic signed [WIDTH:0] V_new;    // One bit wider for overflow protection

    // Power-of-2 limits based on WIDTH
    localparam signed [WIDTH-1:0] MAX_LIMIT =  (2**(WIDTH-1)) - 1;
    localparam signed [WIDTH-1:0] MIN_LIMIT = -(2**(WIDTH-1));

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            V_mem <= 0;
            spike <= 0;
            refr_count <= 0;

            // Set default parameter values
            V_threshold <= 30000;
            leak_factor <= 100;
            refr_period <= 10;
            V_max <= MAX_LIMIT;
            V_min <= MIN_LIMIT;

        end else if (load_params) begin
            // Ensure V_max and V_min are within valid power-of-2 range
            logic signed [WIDTH-1:0] temp_V_max, temp_V_min;
            
            temp_V_max = (new_V_max > MAX_LIMIT) ? MAX_LIMIT : new_V_max;
            temp_V_min = (new_V_min < MIN_LIMIT) ? MIN_LIMIT : new_V_min;

            // Ensure V_max is always greater than V_min
            if (temp_V_max < temp_V_min) begin
                V_max <= temp_V_min + 1; // Adjust max to be above min
                V_min <= temp_V_min;
            end else begin
                V_max <= temp_V_max;
                V_min <= temp_V_min;
            end

            V_threshold <= new_V_threshold;
            leak_factor <= new_leak_factor;
            refr_period <= new_refr_period;

        end else begin
            if (refr_count > 0) begin
                // Refractory period countdown
                refr_count <= refr_count - 1;
                spike <= 0;
            end else begin
                // Compute new potential in extended bit-width
                V_new = V_mem + I_in - leak_factor;

                // Apply saturation limits
                if (V_new > V_max) begin
                    V_mem <= V_max;
                end else if (V_new < V_min) begin
                    V_mem <= V_min;
                end else begin
                    V_mem <= V_new[WIDTH-1:0]; // Assign back to correct width
                end

                // Check for spike
                if (V_mem >= V_threshold) begin
                    spike <= 1;
                    V_mem <= 0; // Reset potential after spike
                    refr_count <= refr_period - 1; // Start refractory period
                end else begin
                    spike <= 0;
                end
            end
        end
    end
endmodule
