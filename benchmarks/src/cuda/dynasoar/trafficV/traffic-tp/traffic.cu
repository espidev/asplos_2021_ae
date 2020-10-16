#include "traffic.h"

static const int kNumBlockSize = 256;

static const char kCellTypeNormal = 1;
static const char kCellTypeProducer = 2;

using IndexT = int;
using CellPointerT = IndexT;
#include "../dataset.h"


__managed__ CellBase **dev_cells;

// Need 2 arrays of both, so we can swap.
__device__ int *d_Car_active;
__device__ int *d_Car_active_2;
__managed__ CarBase **dev_cars;
__managed__ CarBase **dev_cars_2;

// For prefix sum array compaction.
__device__ int *d_prefix_sum_temp;
__device__ int *d_prefix_sum_output;
int *h_prefix_sum_temp;
int *h_prefix_sum_output;
int *h_Car_active;
int *h_Car_active_2;

__device__ int d_num_cells;
__device__ int d_num_cars;
__device__ int d_num_cars_2;
int host_num_cells;
int host_num_cars;

// TODO: Consider migrating to SoaAlloc.
TrafficLight *h_traffic_lights;
__managed__ TrafficLightBase **d_traffic_lights;

// Only for rendering.
__device__ int dev_num_cells;
__device__ float *dev_Cell_pos_x;
__device__ float *dev_Cell_pos_y;
__device__ bool *dev_Cell_occupied;
float *host_Cell_pos_x;
float *host_Cell_pos_y;
bool *host_Cell_occupied;
float *host_data_Cell_pos_x;
float *host_data_Cell_pos_y;
bool *host_data_Cell_occupied;

// coal ready
__device__ void Car_step_extend_path(IndexT self) {
    void **vtable;
    //if(self == 0 ) printf("-----------$self %p \n",dev_cars[self]);
    
    COAL_CarBase_get_position(dev_cars[self]);
    CellBase *cell = CLEANPTR( dev_cars[self] ,CarBase *)->get_position();
    
    CellBase *next_cell;
    COAL_CarBase_get_velocity(dev_cars[self]);
    for (int i = 0; i < CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity(); ++i) {
       
        COAL_CellBase_get_is_target(cell);
        bool cond = CLEANPTR( cell ,CellBase *)->get_is_target();
        COAL_CellBase_is_sink(cell);
        if (CLEANPTR( cell ,CellBase *)->is_sink() || cond) {
            break;
        }

      
        COAL_CarBase_next_step(dev_cars[self]);
        next_cell = CLEANPTR( dev_cars[self] ,CarBase *)->next_step(cell);
        assert(next_cell != cell);
       // printf ("hiiiii %p %d\n",next_cell, self);
        COAL_CellBase_is_free(next_cell);
        if (!(CLEANPTR( next_cell ,CellBase *)->is_free())) break;
        
        cell = next_cell;
        COAL_CarBase_set_path(dev_cars[self]);
        CLEANPTR( dev_cars[self] ,CarBase *)->set_path(cell, i);
        COAL_CarBase_get_path_length(dev_cars[self]);
        int path_len = CLEANPTR( dev_cars[self] ,CarBase *)->get_path_length();
        COAL_CarBase_set_path_length(dev_cars[self]);
        CLEANPTR( dev_cars[self] ,CarBase *)->set_path_length(path_len + 1);
    }
    COAL_CarBase_get_path_length(dev_cars[self]);
    int path_len = CLEANPTR( dev_cars[self] ,CarBase *)->get_path_length();
    COAL_CarBase_set_velocity(dev_cars[self]);
    CLEANPTR( dev_cars[self] ,CarBase *)->set_velocity(path_len);
}

__device__ void Car_step_constraint_velocity(IndexT self) {
    void **vtable;
    // This is actually only needed for the very first iteration, because a car
    // may be positioned on a traffic light cell.
    COAL_CarBase_get_velocity(dev_cars[self]);
    int vel = CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity();
    COAL_CarBase_get_position(dev_cars[self]);
    CellBase *cell = CLEANPTR( dev_cars[self] ,CarBase *)->get_position();
    COAL_CellBase_get_current_max_velocity(cell);
    if (vel > CLEANPTR( cell ,CellBase *)->get_current_max_velocity()) {
        COAL_CellBase_get_current_max_velocity(cell);
        int max_velocity = CLEANPTR( cell ,CellBase *)->get_current_max_velocity();
        COAL_CarBase_set_velocity(dev_cars[self]);
        CLEANPTR( dev_cars[self] ,CarBase *)->set_velocity(max_velocity);
    }

    int path_index = 0;
    int distance = 1;
    COAL_CarBase_get_velocity(dev_cars[self]);
    while (distance <= CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity()) {
        // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
        //            is allowed.
        // Now check if next cell can be entered.
        COAL_CarBase_get_path(dev_cars[self]);
        CellBase *next_cell = CLEANPTR( dev_cars[self] ,CarBase *)->get_path(path_index);
        COAL_CellBase_is_free(next_cell);
        // Avoid collision.
        if (!CLEANPTR( next_cell ,CellBase *)->is_free()) {
            // Cannot enter cell.
            --distance;
            COAL_CarBase_set_velocity(dev_cars[self]);
            CLEANPTR( dev_cars[self] ,CarBase *)->set_velocity(distance);
            break;
        }  // else: Can enter next cell.
        COAL_CarBase_get_velocity(dev_cars[self]);
        int curr_vel = CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity();
        COAL_CellBase_get_current_max_velocity(next_cell);
        if (curr_vel > CLEANPTR( next_cell ,CellBase *)->get_current_max_velocity()) {
            // Car is too fast for this cell.
            COAL_CellBase_get_current_max_velocity(next_cell);
            if (CLEANPTR( next_cell ,CellBase *)->get_current_max_velocity() > distance - 1) {
                // Even if we slow down, we would still make progress.
                COAL_CellBase_get_current_max_velocity(next_cell);
                int max = CLEANPTR( next_cell ,CellBase *)->get_current_max_velocity();
                COAL_CarBase_set_velocity(dev_cars[self]);
                CLEANPTR( dev_cars[self] ,CarBase *)->set_velocity(max);
            } else {
                // Do not enter the next cell.
                --distance;
                assert(distance >= 0);
                COAL_CarBase_set_velocity(dev_cars[self]);
                CLEANPTR( dev_cars[self] ,CarBase *)->set_velocity(distance);
                break;
            }
        }

        ++distance;
        ++path_index;
    }

    --distance;

#ifndef NDEBUG
    COAL_CarBase_get_velocity(dev_cars[self]);
    for (int i = 0; i < CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity(); ++i) {
        COAL_CarBase_get_path(dev_cars[self]);
        CellBase *path = CLEANPTR( dev_cars[self] ,CarBase *)->get_path(i);
        COAL_CellBase_is_free(path);
        assert(CLEANPTR( path ,CellBase *)->is_free());
        COAL_CarBase_get_path(dev_cars[self]);
        assert(i == 0 || CLEANPTR( dev_cars[self] ,CarBase *)->get_path(i - 1) != path);
    }
    // TODO: Check why the cast is necessary.
    COAL_CarBase_get_velocity(dev_cars[self]);
    assert(distance <= CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity());
#endif  // NDEBUG
}

__device__ void Car_step_move(IndexT self) {
    void **vtable;
    COAL_CarBase_get_position(dev_cars[self]);
    CellBase *cell = CLEANPTR( dev_cars[self] ,CarBase *)->get_position();
    COAL_CarBase_get_velocity(dev_cars[self]);
    for (int i = 0; i < CLEANPTR( dev_cars[self] ,CarBase *)->get_velocity(); ++i) {
        COAL_CarBase_get_path(dev_cars[self]);
        assert(CLEANPTR( dev_cars[self] ,CarBase *)->get_path(i) != cell);
        COAL_CarBase_get_path(dev_cars[self]);
        cell = CLEANPTR( dev_cars[self] ,CarBase *)->get_path(i);
        COAL_CellBase_is_free(cell);
        assert(CLEANPTR( cell ,CellBase *)->is_free());
        COAL_CarBase_get_position(dev_cars[self]);
        CellBase *ptr = CLEANPTR( dev_cars[self] ,CarBase *)->get_position();
        COAL_CellBase_release(ptr);
        CLEANPTR( ptr ,CellBase *)->release();
        COAL_CellBase_occupy(cell);
        CLEANPTR( cell ,CellBase *)->occupy(dev_cars[self]);
        COAL_CarBase_set_position(dev_cars[self]);
        CLEANPTR( dev_cars[self] ,CarBase *)->set_position(cell);
    }
    COAL_CarBase_get_position(dev_cars[self]);
    CellBase *ptr = CLEANPTR( dev_cars[self] ,CarBase *)->get_position();
    COAL_CellBase_is_sink(ptr);
    bool cond = CLEANPTR( ptr ,CellBase *)->is_sink();
    COAL_CellBase_get_is_target(ptr);
    if (cond || CLEANPTR( ptr ,CellBase *)->get_is_target()) {
        // Remove car from the simulation. Will be added again in the next
        // iteration.
        COAL_CellBase_release(ptr);
        CLEANPTR( ptr ,CellBase *)->release();
        COAL_CarBase_set_position(dev_cars[self]);
        CLEANPTR( dev_cars[self] ,CarBase *)->set_position(nullptr);
        d_Car_active[self] = 0;
    }
}

__device__ void Car_step_slow_down(IndexT self) {
    void **vtable;
    // 20% change of slowdown.
    CarBase *ptr = dev_cars[self];
    COAL_CarBase_get_velocity(ptr);
    int vel = CLEANPTR( ptr ,CarBase *)->get_velocity();
    COAL_CarBase_random_uni(ptr);
    if (CLEANPTR( ptr ,CarBase *)->random_uni() < 0.2 && vel > 0) {
        COAL_CarBase_set_velocity(ptr);
        CLEANPTR( ptr ,CarBase *)->set_velocity(vel - 1);
    }
}

__device__ IndexT new_Car(int seed, IndexT cell, int max_velocity) {
    void **vtable;
    IndexT idx = atomicAdd(&d_num_cars, 1);
    assert(idx >= 0 && idx < kMaxNumCars);
    CarBase *ptr = dev_cars[idx];
    assert(!d_Car_active[idx]);
    COAL_CarBase_set_position(ptr);
    CLEANPTR( ptr ,CarBase *)->set_position(dev_cells[cell]);
    COAL_CarBase_set_path_length(ptr);
    CLEANPTR( ptr ,CarBase *)->set_path_length(0);

    COAL_CarBase_set_velocity(ptr);

    CLEANPTR( ptr ,CarBase *)->set_velocity(0);
    COAL_CarBase_set_max_velocity(ptr);
    CLEANPTR( ptr ,CarBase *)->set_max_velocity(max_velocity);
    d_Car_active[idx] = 1;

    COAL_CellBase_is_free(dev_cells[cell]);
    assert(CLEANPTR( dev_cells[cell] ,CellBase *)->is_free());
    COAL_CellBase_occupy(dev_cells[cell]);
    CLEANPTR( dev_cells[cell] ,CellBase *)->occupy(dev_cars[idx]);
    curand_init(seed, 0, 0, &CLEANPTR( ptr ,CarBase *)->random_state);

    return idx;
}

__device__ void ProducerCell_create_car(IndexT self) {
    void **vtable;
    assert(CLEANPTR( dev_cells[self] ,CellBase *)->type == kCellTypeProducer);
    COAL_CellBase_is_free(dev_cells[self]);
    if (CLEANPTR( dev_cells[self] ,CellBase *)->is_free()) {
        float r = curand_uniform(&CLEANPTR( dev_cells[self] ,CellBase *)->random_state);
        if (r < kCarAllocationRatio) {
            IndexT new_car = new_Car(
                /*seed=*/curand(&CLEANPTR( dev_cells[self] ,CellBase *)->random_state), /*cell=*/self,
                /*max_velocity=*/curand(&CLEANPTR( dev_cells[self] ,CellBase *)->random_state) %
                        (kMaxVelocity / 2) +
                    kMaxVelocity / 2);
        }
    }
}

__device__ IndexT new_Cell(int max_velocity, float x, float y) {
    IndexT idx = atomicAdd(&d_num_cells, 1);

    CLEANPTR(dev_cells[idx],CellBase *)->car = nullptr;
    CLEANPTR(dev_cells[idx],CellBase *)->max_velocity = max_velocity;
    CLEANPTR(dev_cells[idx],CellBase *)->current_max_velocity = max_velocity;
    CLEANPTR(dev_cells[idx],CellBase *)->num_incoming = 0;
    CLEANPTR(dev_cells[idx],CellBase *)->num_outgoing = 0;
    CLEANPTR(dev_cells[idx],CellBase *)->x = x;
    CLEANPTR(dev_cells[idx],CellBase *)->y = y;
    CLEANPTR(dev_cells[idx],CellBase *)->is_target = false;
    CLEANPTR(dev_cells[idx],CellBase *)->type = kCellTypeNormal;

    return idx;
}

__device__ IndexT new_ProducerCell(int max_velocity, float x, float y,
                                   int seed) {
    IndexT idx = new_Cell(max_velocity, x, y);
    CLEANPTR(dev_cells[idx],CellBase *)->type = kCellTypeProducer;
    curand_init(seed, 0, 0, &CLEANPTR(dev_cells[idx],CellBase *)->random_state);

    return idx;
}

__global__ void kernel_traffic_light_step() {
    void **vtable;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        TrafficLightBase *ptr = d_traffic_lights[i];
        COAL_TrafficLightBase_get_num_cells(ptr);
        if (CLEANPTR( ptr ,TrafficLightBase  *)->get_num_cells() > 0) {
            COAL_TrafficLightBase_get_timer(ptr);
            int timer = CLEANPTR( ptr ,TrafficLightBase  *)->get_timer();
            COAL_TrafficLightBase_get_phase_time(ptr);
            int phase_time = CLEANPTR( ptr ,TrafficLightBase  *)->get_phase_time();
            COAL_TrafficLightBase_set_timer(ptr);
            CLEANPTR( ptr ,TrafficLightBase  *)->set_timer((timer + 1) % phase_time);
            COAL_TrafficLightBase_get_timer(ptr);
            if (CLEANPTR( ptr ,TrafficLightBase  *)->get_timer() == 0) {
                COAL_TrafficLightBase_get_phase(ptr);
                int phase = CLEANPTR( ptr ,TrafficLightBase  *)->get_phase();
                COAL_TrafficLightBase_get_cell(ptr);
                assert(CLEANPTR( ptr ,TrafficLightBase  *)->get_cell(phase) != nullptr);
                COAL_TrafficLightBase_get_phase(ptr);
                phase = CLEANPTR( ptr ,TrafficLightBase  *)->get_phase();
                COAL_TrafficLightBase_get_cell(ptr);
                CellBase *ptr2 = CLEANPTR( ptr ,TrafficLightBase  *)->get_cell(phase);
                COAL_CellBase_set_current_max_velocity(ptr2);
                CLEANPTR( ptr2 ,CellBase *)->set_current_max_velocity(0);
                COAL_TrafficLightBase_get_phase(ptr);
                int phase_2 = CLEANPTR( ptr ,TrafficLightBase  *)->get_phase();
                COAL_TrafficLightBase_get_num_cells(ptr);
                int num_cells = CLEANPTR( ptr ,TrafficLightBase  *)->get_num_cells();
                COAL_TrafficLightBase_set_phase(ptr);
                CLEANPTR( ptr ,TrafficLightBase  *)->set_phase((phase_2 + 1) % num_cells);
                COAL_TrafficLightBase_get_phase(ptr);
                phase_2 = CLEANPTR( ptr ,TrafficLightBase  *)->get_phase();
                COAL_TrafficLightBase_get_cell(ptr);
                ptr2 = CLEANPTR( ptr ,TrafficLightBase  *)->get_cell(phase_2);
                COAL_CellBase_remove_speed_limit(ptr2);
                CLEANPTR( ptr2 ,CellBase *)->remove_speed_limit();
            }
        }
        // d_traffic_lights[i]->step();
    }
}

__global__ void kernel_create_nodes() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        curandState_t state;
        curand_init(i, 0, 0, &state);

        assert(d_nodes[i].x >= 0 && d_nodes[i].x <= 1);
        assert(d_nodes[i].y >= 0 && d_nodes[i].y <= 1);

        for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
            d_nodes[i].cell_out[j] = new_Cell(
                /*max_velocity=*/curand(&state) % (kMaxVelocity / 2) +
                    kMaxVelocity / 2,
                d_nodes[i].x, d_nodes[i].y);
        }
    }
}

__device__ IndexT connect_intersections(IndexT from, Node *target,
                                        int incoming_idx,
                                        curandState_t &state) {
    // Create edge.
    float dx = target->x - CLEANPTR( dev_cells[from] ,CellBase *)->x;
    float dy = target->y - CLEANPTR( dev_cells[from] ,CellBase *)->y;
    float dist = sqrt(dx * dx + dy * dy);
    int steps = dist / kCellLength;
    float step_x = dx / steps;
    float step_y = dy / steps;
    IndexT prev = from;

    for (int j = 0; j < steps; ++j) {
        float new_x = CLEANPTR( dev_cells[from] ,CellBase *)->x + j * step_x;
        float new_y = CLEANPTR( dev_cells[from] ,CellBase *)->y + j * step_y;
        assert(new_x >= 0 && new_x <= 1);
        assert(new_y >= 0 && new_y <= 1);
        IndexT next;

        if (curand_uniform(&state) < kProducerRatio) {
            next = new_ProducerCell(CLEANPTR( dev_cells[prev] ,CellBase *)->max_velocity, new_x, new_y,
                                    curand(&state));
        } else {
            next = new_Cell(CLEANPTR( dev_cells[prev] ,CellBase *)->max_velocity, new_x, new_y);
        }

        if (curand_uniform(&state) < kTargetRatio) {
            CLEANPTR( dev_cells[next] ,CellBase *)->set_target();
        }

        CLEANPTR( dev_cells[prev] ,CellBase *)->set_num_outgoing(1);
        CLEANPTR( dev_cells[prev] ,CellBase *)->set_outgoing(0,dev_cells[next] );

        CLEANPTR( dev_cells[next] ,CellBase *)->set_num_incoming(1);
        CLEANPTR( dev_cells[next] ,CellBase *)->set_incoming(0,  dev_cells[prev]);

        prev = next;
    }

    // Connect to all outgoing nodes of target.
    CLEANPTR( dev_cells[prev] ,CellBase *)->set_num_outgoing(target->num_outgoing);
    for (int i = 0; i < target->num_outgoing; ++i) {
        IndexT next = target->cell_out[i];
        // num_incoming set later.
        CLEANPTR( dev_cells[prev] ,CellBase *)->set_outgoing(i,  dev_cells[next] );
        CLEANPTR( dev_cells[next] ,CellBase *)->set_incoming(incoming_idx,  dev_cells[prev] );
    }

    return prev;
}

__global__ void kernel_create_edges() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        curandState_t state;
        curand_init(i, 0, 0, &state);

        for (int k = 0; k < d_nodes[i].num_outgoing; ++k) {
            int target = d_nodes[i].node_out[k];
            int target_pos = d_nodes[i].node_out_pos[k];

            IndexT last = connect_intersections(
                d_nodes[i].cell_out[k], &d_nodes[target], target_pos, state);

            CLEANPTR(dev_cells[last],CellBase *)->set_current_max_velocity(0);
            d_nodes[target].cell_in[target_pos] = last;
        }
    }
}

__global__ void kernel_create_traffic_lights() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        new (CLEANPTR(d_traffic_lights[i], TrafficLightBase *)) TrafficLight(
            /*num_cells=*/d_nodes[i].num_incoming,
            /*phase_time=*/5);

        for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
            CLEANPTR( dev_cells[d_nodes[i].cell_out[j]],CellBase *) ->set_num_incoming(
                d_nodes[i].num_incoming);
        }

        for (int j = 0; j < d_nodes[i].num_incoming; ++j) {
            CLEANPTR(d_traffic_lights[i], TrafficLightBase *)->set_cell(j, dev_cells[d_nodes[i].cell_in[j]]);
            CLEANPTR(dev_cells[d_nodes[i].cell_in[j]],CellBase *)->set_current_max_velocity(
                0);  // Set to "red".
        }
    }
}

template <class Type, class TypeBase>
__global__ void device_alloc(TypeBase **ptr, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        ptr[i] = new Type();
        assert(ptr[i] != nullptr);
    }
}
void create_street_network(obj_alloc *alloc) {
    int zero = 0;
    cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMalloc(&h_nodes, sizeof(Node) * kNumIntersections);
    cudaMemcpyToSymbol(d_nodes, &h_nodes, sizeof(Node *), 0,
                       cudaMemcpyHostToDevice);
    // cudaMalloc(&d_traffic_lights, sizeof(TrafficLight *) *
    // kNumIntersections);

    // device_alloc<TrafficLight, TrafficLightBase>
    //     <<<(kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
    //        kNumBlockSize>>>(d_traffic_lights, kNumIntersections);

    // gpuErrchk(cudaDeviceSynchronize());
    d_traffic_lights = (TrafficLightBase **)alloc->calloc<TrafficLightBase *>(
        kNumIntersections);
    for (int i = 0; i < kNumIntersections; i += 1) {
        d_traffic_lights[i] = (TrafficLightBase *)alloc->my_new<TrafficLight>();
    }
    alloc->toDevice();
    // Create basic structure on host.
    create_network_structure();

    kernel_create_nodes<<<(kNumIntersections + kNumBlockSize - 1) /
                              kNumBlockSize,
                          kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_create_edges<<<(kNumIntersections + kNumBlockSize - 1) /
                              kNumBlockSize,
                          kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_create_traffic_lights<<<(kNumIntersections + kNumBlockSize - 1) /
                                       kNumBlockSize,
                                   kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    // Allocate helper data structures for rendering.
    cudaMemcpyFromSymbol(&host_num_cells, d_num_cells, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
    cudaMalloc(&host_Cell_pos_x, sizeof(float) * host_num_cells);
    cudaMemcpyToSymbol(dev_Cell_pos_x, &host_Cell_pos_x, sizeof(float *), 0,
                       cudaMemcpyHostToDevice);
    cudaMalloc(&host_Cell_pos_y, sizeof(float) * host_num_cells);
    cudaMemcpyToSymbol(dev_Cell_pos_y, &host_Cell_pos_y, sizeof(float *), 0,
                       cudaMemcpyHostToDevice);
    cudaMalloc(&host_Cell_occupied, sizeof(bool) * host_num_cells);
    cudaMemcpyToSymbol(dev_Cell_occupied, &host_Cell_occupied, sizeof(bool *),
                       0, cudaMemcpyHostToDevice);
    host_data_Cell_pos_x = (float *)malloc(sizeof(float) * host_num_cells);
    host_data_Cell_pos_y = (float *)malloc(sizeof(float) * host_num_cells);
    host_data_Cell_occupied = (bool *)malloc(sizeof(bool) * host_num_cells);

#ifndef NDEBUG
    printf("Number of cells: %i\n", host_num_cells);
#endif  // NDEBUG
}

void step_traffic_lights() {
    // TODO: Consider migrating this to SoaAlloc.
    kernel_traffic_light_step<<<(kNumIntersections + kNumBlockSize - 1) /
                                    kNumBlockSize,
                                kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());
}

__device__ void Cell_add_to_rendering_array(IndexT self) {
    int idx = atomicAdd(&dev_num_cells, 1);
    dev_Cell_pos_x[idx] = dev_cells[self]->x;
    dev_Cell_pos_y[idx] = dev_cells[self]->y;
    dev_Cell_occupied[idx] = !dev_cells[self]->is_free();
}

__global__ void kernel_Cell_add_to_rendering_array() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cells;
         i += blockDim.x * gridDim.x) {
        Cell_add_to_rendering_array(i);
    }
}

void transfer_data() {
    int zero = 0;
    cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);

    kernel_Cell_add_to_rendering_array<<<(host_num_cells + kNumBlockSize - 1) /
                                             kNumBlockSize,
                                         kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(host_data_Cell_pos_x, host_Cell_pos_x,
               sizeof(float) * host_num_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_Cell_pos_y, host_Cell_pos_y,
               sizeof(float) * host_num_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_Cell_occupied, host_Cell_occupied,
               sizeof(bool) * host_num_cells, cudaMemcpyDeviceToHost);

    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void kernel_ProducerCell_create_car() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cells;
         i += blockDim.x * gridDim.x) {
        if (CLEANPTR( dev_cells[i] ,CellBase *)->type == kCellTypeProducer) {
            ProducerCell_create_car(i);
        }
    }
}

__device__ void Car_step_prepare_path(IndexT self) {
    void **vtable;
    COAL_CarBase_step_initialize_iteration(dev_cars[self]);
    CLEANPTR( dev_cars[self] ,CarBase *)->step_initialize_iteration();
    COAL_CarBase_step_accelerate(dev_cars[self]);
    CLEANPTR( dev_cars[self] ,CarBase *)->step_accelerate();


    Car_step_extend_path(self);
    
    Car_step_constraint_velocity(self);
    Car_step_slow_down(self);
}

__global__ void kernel_Car_step_prepare_path() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
         i += blockDim.x * gridDim.x) {
        if (d_Car_active[i]) {
            Car_step_prepare_path(i);
        }
    }
}

__global__ void kernel_fill_car_indices() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
         i += blockDim.x * gridDim.x) {
        d_Car_active[i] = 0;
        d_Car_active_2[i] = 0;
    }
}

__global__ void kernel_Car_step_move() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
         i += blockDim.x * gridDim.x) {
        if (d_Car_active[i]) {
            Car_step_move(i);
        }
    }
}

__device__ int d_checksum;
__global__ void kernel_compute_checksum() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
         i += blockDim.x * gridDim.x) {
        if (d_Car_active[i]) {
            atomicAdd(&d_checksum, 1);
        }
    }
}

int checksum() {
    int zero = 0;
    cudaMemcpyToSymbol(d_checksum, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    kernel_compute_checksum<<<128, 128>>>();

    int result;
    cudaMemcpyFromSymbol(&result, d_checksum, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
    return result;
}

void step() {
    kernel_ProducerCell_create_car<<<(host_num_cells + kNumBlockSize - 1) /
                                         kNumBlockSize,
                                     kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpyFromSymbol(&host_num_cars, d_num_cars, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
    step_traffic_lights();

    kernel_Car_step_prepare_path<<<
        (host_num_cars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_Car_step_move<<<(host_num_cars + kNumBlockSize - 1) / kNumBlockSize,
                           kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());
}

void allocate_memory(obj_alloc *alloc) {
    // cudaMalloc(&dev_cells, sizeof(Cell *) * kMaxNumCells);
    dev_cells = (CellBase **)alloc->calloc<CellBase *>(kMaxNumCells);
    for (int i = 0; i < kMaxNumCells; i += 1) {
        dev_cells[i] = (CellBase *)alloc->my_new<Cell>();
    }

    for (int i = 0; i < kMaxNumCells; i += 1) {
        (dummy *)alloc->my_new<dummy>();
    }
    // device_alloc<Cell, CellBase>
    //     <<<(kMaxNumCells + kNumBlockSize - 1) / kNumBlockSize,
    //     kNumBlockSize>>>(
    //         dev_cells, kMaxNumCells);

    gpuErrchk(cudaDeviceSynchronize());

    // cudaMalloc(&dev_cars, sizeof(Car *) * kMaxNumCars);
    // cudaMalloc(&dev_cars_2, sizeof(Car *) * kMaxNumCars);
    dev_cars = (CarBase **)alloc->calloc<CarBase *>(kMaxNumCars);
    for (int i = 0; i < kMaxNumCars; i += 1) {
        dev_cars[i] = (CarBase *)alloc->my_new<Car>();
    }
    dev_cars_2 = (CarBase **)alloc->calloc<CarBase *>(kMaxNumCars);
    for (int i = 0; i < kMaxNumCars; i += 1) {
        dev_cars_2[i] = (CarBase *)alloc->my_new<Car>();
    }
    alloc->toDevice();
    // device_alloc<Car, CarBase>
    //     <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize,
    //     kNumBlockSize>>>(
    //         dev_cars, kMaxNumCars);
    // gpuErrchk(cudaDeviceSynchronize());
    // device_alloc<Car>
    //     <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize,
    //     kNumBlockSize>>>(
    //         dev_cars_2, kMaxNumCars);
    gpuErrchk(cudaDeviceSynchronize());
    cudaMalloc(&h_Car_active, sizeof(int) * kMaxNumCars);
    cudaMemcpyToSymbol(d_Car_active, &h_Car_active, sizeof(int *), 0,
                       cudaMemcpyHostToDevice);

    // Car *h_cars_2;
    // cudaMalloc(&h_cars_2, sizeof(Car) * kMaxNumCars);
    // cudaMemcpyToSymbol(dev_cars_2, &h_cars_2, sizeof(Car *), 0,
    //                    cudaMemcpyHostToDevice);

    cudaMalloc(&h_Car_active_2, sizeof(int) * kMaxNumCars);
    cudaMemcpyToSymbol(d_Car_active_2, &h_Car_active_2, sizeof(int *), 0,
                       cudaMemcpyHostToDevice);

    cudaMalloc(&h_prefix_sum_temp, 3 * sizeof(int) * kMaxNumCars);
    cudaMemcpyToSymbol(d_prefix_sum_temp, &h_prefix_sum_temp, sizeof(int *), 0,
                       cudaMemcpyHostToDevice);

    cudaMalloc(&h_prefix_sum_output, sizeof(int) * kMaxNumCars);
    cudaMemcpyToSymbol(d_prefix_sum_output, &h_prefix_sum_output, sizeof(int *),
                       0, cudaMemcpyHostToDevice);

    kernel_fill_car_indices<<<128, 128>>>();
    gpuErrchk(cudaDeviceSynchronize());

    int zero = 0;
    cudaMemcpyToSymbol(d_num_cells, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_num_cars, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
}

__global__ void kernel_compact_initialize() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kMaxNumCars;
         i += blockDim.x * gridDim.x) {
        d_Car_active_2[i] = 0;
    }
}

__global__ void kernel_compact_cars() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
         i += blockDim.x * gridDim.x) {
        if (d_Car_active[i]) {
            int target = d_prefix_sum_output[i];

            // Copy i --> target.
            // dev_cars_2[target] = dev_cars[i];
            memcpy(CLEANPTR( dev_cars_2[target] ,CarBase *), CLEANPTR( dev_cars[i] ,CarBase *), sizeof(Car));
            d_Car_active_2[target] = 1;

            // Update pointer in Cell.
            CLEANPTR(CLEANPTR( dev_cars[i] ,CarBase *)->position,CellBase *)->car = dev_cars[target];

            atomicAdd(&d_num_cars_2, 1);
        }
    }
}

__global__ void kernel_compact_swap_pointers() {
    {
        auto *tmp = dev_cars;
        dev_cars = dev_cars_2;
        dev_cars_2 = tmp;
    }

    {
        auto *tmp = d_Car_active;
        d_Car_active = d_Car_active_2;
        d_Car_active_2 = tmp;
    }

    d_num_cars = d_num_cars_2;
}

void compact_car_array() {
    int zero = 0;
    cudaMemcpyToSymbol(d_num_cars_2, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyFromSymbol(&host_num_cars, d_num_cars, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);

    // TODO: Prefix sum broken for num_objects < 256.
    auto prefix_sum_size = host_num_cars < 256 ? 256 : host_num_cars;
    size_t temp_size = 3 * kMaxNumCars;
    cub::DeviceScan::ExclusiveSum(h_prefix_sum_temp, temp_size, h_Car_active,
                                  h_prefix_sum_output, prefix_sum_size);
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_initialize<<<
        (kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_cars<<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize,
                          kNumBlockSize>>>();
    gpuErrchk(cudaDeviceSynchronize());

    kernel_compact_swap_pointers<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());

    auto *tmp = h_Car_active;
    h_Car_active = h_Car_active_2;
    h_Car_active_2 = tmp;
}
void init(obj_alloc *alloc) {
    allocate_memory(alloc);

    create_street_network(alloc);
}
int main(int /*argc*/, char ** argv) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
    mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
    obj_alloc my_obj_alloc(&shared_mem, atoll(argv[1]));
    init(&my_obj_alloc);
    my_obj_alloc.toDevice();
    printf("mem alloc done\n");
    my_obj_alloc.create_table();
    vfun_table = my_obj_alloc.get_vfun_table();
    auto time_start = std::chrono::system_clock::now();

    for (int i = 0; i < kNumIterations; ++i) {
        //printf("%d\n",i);
        step();

        compact_car_array();
    }

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto millis =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

#ifndef NDEBUG
    printf("Checksum: %i\n", checksum());
#endif  // NDEBUG

    printf("%lu\n", millis);
}