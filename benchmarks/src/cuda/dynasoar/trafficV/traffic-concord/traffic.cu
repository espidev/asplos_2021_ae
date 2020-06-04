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

__device__ void Car_step_extend_path(IndexT self) {
    // CONCORD
    CellBase *cell;
    CONCORD(cell, dev_cars[self], get_position());

    CellBase *next_cell;

    // CONCORD
    int vel;
    CONCORD(vel, dev_cars[self], get_velocity());
    for (int i = 0; i < vel; ++i) {
        // CONCORD
        bool cond;
        CONCORD(cond, cell, get_is_target());
        ;
        // CONCORD
        bool cond2;
        CONCORD(cond2, cell, is_sink());
        if (cond2 || cond) {
            break;
        }

        // CONCORD
        CONCORD(next_cell, dev_cars[self], next_step(cell));

        assert(next_cell != cell);

        // CONCORD
        bool cond3;
        CONCORD(cond3, next_cell, is_free());
        if (!cond3) break;

        cell = next_cell;
        // CONCORD
        CONCORD(dev_cars[self], set_path(cell, i));
        ;
        // CONCORD
        int path_len;
        CONCORD(path_len, dev_cars[self], get_path_length());
        ;
        // CONCORD
        CONCORD(dev_cars[self], set_path_length(path_len + 1));
        ;
    }
    // CONCORD
    int path_len;
    CONCORD(path_len, dev_cars[self], get_path_length());
    ;
    // CONCORD
    CONCORD(dev_cars[self], set_velocity(path_len));
    ;
}

__device__ void Car_step_constraint_velocity(IndexT self) {
    // This is actually only needed for the very first iteration, because a car
    // may be positioned on a traffic light cell.
    // CONCORD
    int vel;
    CONCORD(vel, dev_cars[self], get_velocity());

    // CONCORD
    CellBase *cell;
    CONCORD(cell, dev_cars[self], get_position());
    int vel_max;
    CONCORD(vel_max, cell, get_current_max_velocity());
    // CONCORD
    if (vel > vel_max) {
        // CONCORD
        int max_velocity;
        CONCORD(max_velocity, cell, get_current_max_velocity());
        ;
        // CONCORD
        CONCORD(dev_cars[self], set_velocity(max_velocity));
        ;
    }

    int path_index = 0;
    int distance = 1;

    // CONCORD
    int vel3;
    CONCORD(vel3, dev_cars[self], get_velocity());
    while (distance <= vel3) {
        // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
        //            is allowed.
        // Now check if next cell can be entered.
        // CONCORD
        CellBase *next_cell;
        CONCORD(next_cell, dev_cars[self], get_path(path_index));
        ;

        // Avoid collision.
        // CONCORD
        bool cond4;
        CONCORD(cond4, next_cell, is_free());
        if (!cond4) {
            // Cannot enter cell.
            --distance;
            // CONCORD
            CONCORD(dev_cars[self], set_velocity(distance));
            ;
            break;
        }  // else: Can enter next cell.

        // CONCORD
        int curr_vel;
        CONCORD(curr_vel, dev_cars[self], get_velocity());
        ;

        // CONCORD
        int cur_max;
        CONCORD(cur_max, next_cell, get_current_max_velocity());
        if (curr_vel > cur_max) {
            // Car is too fast for this cell.
            // CONCORD
            int cur_max2;
            CONCORD(cur_max2, next_cell, get_current_max_velocity());
            if (cur_max2 > distance - 1) {
                // Even if we slow down, we would still make progress.
                // CONCORD
                int max;
                CONCORD(max, next_cell, get_current_max_velocity());

                // CONCORD
                CONCORD(dev_cars[self], set_velocity(max));

            } else {
                // Do not enter the next cell.
                --distance;
                assert(distance >= 0);

                // CONCORD
                CONCORD(dev_cars[self], set_velocity(distance));

                break;
            }
        }

        ++distance;
        ++path_index;
    }

    --distance;

#ifndef NDEBUG
    // CONCORD
    int aavel;
    CONCORD(aavel, dev_cars[self], get_velocity());
    for (int i = 0; i < aavel; ++i) {
        // CONCORD
        CellBase *path;
        CellBase *pathi;
        CellBase *pathi_1;
        bool cond_free;
        CONCORD(path, dev_cars[self], get_path(i));
        CONCORD(cond_free, path, is_free());
        assert(cond_free);
        CONCORD(pathi, dev_cars[self], get_path(i));
        CONCORD(pathi_1, dev_cars[self], get_path(i - 1));
        assert(i == 0 || pathi_1 != pathi);
    }
    // TODO: Check why the cast is necessary.
    // CONCORD
    int ver_;
    CONCORD(ver_, dev_cars[self], get_velocity());
    //assert(distance <= ver_);
#endif  // NDEBUG
}

__device__ void Car_step_move(IndexT self) {
    // CONCORD
    CellBase *cell;
    CONCORD(cell, dev_cars[self], get_position());

    // CONCORD
    int vel;
    CONCORD(vel, dev_cars[self], get_velocity());
    for (int i = 0; i < vel; ++i) {
        // CONCORD
        CellBase *path;
        CONCORD(path, dev_cars[self], get_path(i));
        assert(path != cell);

        // CONCORD
        CONCORD(cell, dev_cars[self], get_path(i));

        // CONCORD
        bool cond21;
        CONCORD(cond21, cell, is_free())
        assert(cond21);
        // CONCORD
        CellBase *ptr;
        CONCORD(ptr, dev_cars[self], get_position());

        // CONCORD
        CONCORD(ptr, release());

        // CONCORD
        CONCORD(cell, occupy(dev_cars[self]));

        // CONCORD
        CONCORD(dev_cars[self], set_position(cell));
    }

    // CONCORD
    CellBase *ptr;
    CONCORD(ptr, dev_cars[self], get_position());
    ;
    // CONCORD
    bool cond;
    CONCORD(cond, ptr, is_sink());
    ;
    // CONCORD
    bool cond32;
    CONCORD(cond32, ptr, get_is_target())
    if (cond || cond32) {
        // Remove car from the simulation. Will be added again in the next
        // iteration.

        // CONCORD
        CONCORD(ptr, release());
        ;
        // CONCORD
        CONCORD(dev_cars[self], set_position(nullptr));
        ;
        d_Car_active[self] = 0;
    }
}

__device__ void Car_step_slow_down(IndexT self) {
    // 20% change of slowdown.
    // CONCORD
    int vel;
    CONCORD(vel, dev_cars[self], get_velocity());
    ;
    // CONCORD
    float rnd;
    CONCORD(rnd, dev_cars[self], random_uni());
    if (rnd < 0.2 && vel > 0) {
        // CONCORD
        CONCORD(dev_cars[self], set_velocity(vel - 1));
        ;
    }
}

__device__ IndexT new_Car(int seed, IndexT cell, int max_velocity) {
    IndexT idx = atomicAdd(&d_num_cars, 1);
    assert(idx >= 0 && idx < kMaxNumCars);

    assert(!d_Car_active[idx]);
    // CONCORD
    CONCORD(dev_cars[idx], set_position(dev_cells[cell]));
    ;
    // CONCORD
    CONCORD(dev_cars[idx], set_path_length(0));
    ;
    // CONCORD
    CONCORD(dev_cars[idx], set_velocity(0));
    ;
    // CONCORD
    CONCORD(dev_cars[idx], set_max_velocity(max_velocity));
    ;
    d_Car_active[idx] = 1;

    // CONCORD
    bool cond;
    CONCORD(cond, dev_cells[cell], is_free())
    assert(cond);
    // CONCORD
    CONCORD(dev_cells[cell], occupy(dev_cars[idx]));
    ;
    curand_init(seed, 0, 0, &dev_cars[idx]->random_state);

    return idx;
}

__device__ void ProducerCell_create_car(IndexT self) {
    assert(dev_cells[self]->type == kCellTypeProducer);
    // CONCORD
    bool cond;
    CONCORD(cond, dev_cells[self], is_free());
    if (cond) {
        float r = curand_uniform(&dev_cells[self]->random_state);
        if (r < kCarAllocationRatio) {
            IndexT new_car = new_Car(
                /*seed=*/curand(&dev_cells[self]->random_state), /*cell=*/self,
                /*max_velocity=*/curand(&dev_cells[self]->random_state) %
                        (kMaxVelocity / 2) +
                    kMaxVelocity / 2);
        }
    }
}

__device__ IndexT new_Cell(int max_velocity, float x, float y) {
    IndexT idx = atomicAdd(&d_num_cells, 1);

    dev_cells[idx]->car = nullptr;
    dev_cells[idx]->max_velocity = max_velocity;
    dev_cells[idx]->current_max_velocity = max_velocity;
    dev_cells[idx]->num_incoming = 0;
    dev_cells[idx]->num_outgoing = 0;
    dev_cells[idx]->x = x;
    dev_cells[idx]->y = y;
    dev_cells[idx]->is_target = false;
    dev_cells[idx]->type = kCellTypeNormal;

    return idx;
}

__device__ IndexT new_ProducerCell(int max_velocity, float x, float y,
                                   int seed) {
    IndexT idx = new_Cell(max_velocity, x, y);
    dev_cells[idx]->type = kCellTypeProducer;
    curand_init(seed, 0, 0, &dev_cells[idx]->random_state);

    return idx;
}

__global__ void kernel_traffic_light_step() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        // CONCORD
        int num_c;
        CONCORD(num_c, d_traffic_lights[i], get_num_cells());
        if (num_c > 0) {
            // CONCORD
            int timer;
            CONCORD(timer, d_traffic_lights[i], get_timer());
            ;
            // CONCORD
            int phase_time;
            CONCORD(phase_time, d_traffic_lights[i], get_phase_time());
            ;
            // CONCORD
            CONCORD(d_traffic_lights[i], set_timer((timer + 1) % phase_time));

            // CONCORD
            if (d_traffic_lights[i]->get_timer() == 0) {
                // CONCORD
                int phase;
                CONCORD(phase, d_traffic_lights[i], get_phase());
                ;
                // CONCORD
                CellBase *ptr22;
                CONCORD(ptr22, d_traffic_lights[i], get_cell(phase));
                assert(ptr22 != nullptr);
                // CONCORD
                CONCORD(phase, d_traffic_lights[i], get_phase());
                ;
                // CONCORD
                CellBase *ptr;
                CONCORD(ptr, d_traffic_lights[i], get_cell(phase));
                ;

                // CONCORD
                CONCORD(ptr, set_current_max_velocity(0));
                ;
                // CONCORD
                int phase_2;
                CONCORD(phase_2, d_traffic_lights[i], get_phase());
                ;
                // CONCORD
                int num_cells;
                CONCORD(num_cells, d_traffic_lights[i], get_num_cells());
                ;
                // CONCORD
                CONCORD(d_traffic_lights[i],
                        set_phase((phase_2 + 1) % num_cells));
                // CONCORD
                CONCORD(phase_2, d_traffic_lights[i], get_phase());
                ;
                // CONCORD
                CONCORD(ptr, d_traffic_lights[i], get_cell(phase_2));
                ;
                // CONCORD
                CONCORD(ptr, remove_speed_limit());
                ;
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
    float dx = target->x - dev_cells[from]->x;
    float dy = target->y - dev_cells[from]->y;
    float dist = sqrt(dx * dx + dy * dy);
    int steps = dist / kCellLength;
    float step_x = dx / steps;
    float step_y = dy / steps;
    IndexT prev = from;

    for (int j = 0; j < steps; ++j) {
        float new_x = dev_cells[from]->x + j * step_x;
        float new_y = dev_cells[from]->y + j * step_y;
        assert(new_x >= 0 && new_x <= 1);
        assert(new_y >= 0 && new_y <= 1);
        IndexT next;

        if (curand_uniform(&state) < kProducerRatio) {
            next = new_ProducerCell(dev_cells[prev]->max_velocity, new_x, new_y,
                                    curand(&state));
        } else {
            next = new_Cell(dev_cells[prev]->max_velocity, new_x, new_y);
        }

        if (curand_uniform(&state) < kTargetRatio) {
            // CONCORD
            CONCORD(dev_cells[next], set_target());
            ;
        }

        // CONCORD
        CONCORD(dev_cells[prev], set_num_outgoing(1));
        ;
        // CONCORD
        CONCORD(dev_cells[prev], set_outgoing(0, dev_cells[next]));
        ;
        // CONCORD
        CONCORD(dev_cells[next], set_num_incoming(1));
        ;
        // CONCORD
        CONCORD(dev_cells[next], set_incoming(0, dev_cells[prev]));
        ;

        prev = next;
    }

    // Connect to all outgoing nodes of target.
    // CONCORD
    CONCORD(dev_cells[prev], set_num_outgoing(target->num_outgoing));
    ;
    for (int i = 0; i < target->num_outgoing; ++i) {
        IndexT next = target->cell_out[i];
        // num_incoming set later.
        // CONCORD
        CONCORD(dev_cells[prev], set_outgoing(i, dev_cells[next]));
        ;
        // CONCORD
        CONCORD(dev_cells[next], set_incoming(incoming_idx, dev_cells[prev]));
        ;
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

            // CONCORD
            CONCORD(dev_cells[last], set_current_max_velocity(0));
            
            d_nodes[target].cell_in[target_pos] = last;
        }
    }
}

__global__ void kernel_create_traffic_lights() {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
         i += blockDim.x * gridDim.x) {
        new (d_traffic_lights[i]) TrafficLight(
            /*num_cells=*/d_nodes[i].num_incoming,
            /*phase_time=*/5);

        for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
            // CONCORD
            CONCORD(dev_cells[d_nodes[i].cell_out[j]],
                    set_num_incoming(d_nodes[i].num_incoming));
        }

        for (int j = 0; j < d_nodes[i].num_incoming; ++j) {
            // CONCORD
            CONCORD(d_traffic_lights[i],
                    set_cell(j, dev_cells[d_nodes[i].cell_in[j]]));
            ;
            // CONCORD
            CONCORD(dev_cells[d_nodes[i].cell_in[j]],
                    set_current_max_velocity(0));
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
void create_street_network() {
    int zero = 0;
    cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                       cudaMemcpyHostToDevice);
    cudaMalloc(&h_nodes, sizeof(Node) * kNumIntersections);
    cudaMemcpyToSymbol(d_nodes, &h_nodes, sizeof(Node *), 0,
                       cudaMemcpyHostToDevice);
    cudaMalloc(&d_traffic_lights, sizeof(TrafficLight *) * kNumIntersections);

    device_alloc<TrafficLight, TrafficLightBase>
        <<<(kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
           kNumBlockSize>>>(d_traffic_lights, kNumIntersections);

    gpuErrchk(cudaDeviceSynchronize());

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
    // CONCORD
    CONCORD(dev_Cell_occupied[idx], !dev_cells[self], is_free());
    ;
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
        if (dev_cells[i]->type == kCellTypeProducer) {
            ProducerCell_create_car(i);
        }
    }
}

__device__ void Car_step_prepare_path(IndexT self) {
    // CONCORD
    CONCORD(dev_cars[self], step_initialize_iteration());
    ;
    // CONCORD
    CONCORD(dev_cars[self], step_accelerate());
    ;
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

void allocate_memory() {
    cudaMalloc(&dev_cells, sizeof(Cell *) * kMaxNumCells);
    device_alloc<Cell, CellBase>
        <<<(kMaxNumCells + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
            dev_cells, kMaxNumCells);
    gpuErrchk(cudaDeviceSynchronize());
    cudaMalloc(&dev_cars, sizeof(Car *) * kMaxNumCars);
    cudaMalloc(&dev_cars_2, sizeof(Car *) * kMaxNumCars);
    device_alloc<Car, CarBase>
        <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
            dev_cars, kMaxNumCars);
    gpuErrchk(cudaDeviceSynchronize());
    device_alloc<Car>
        <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
            dev_cars_2, kMaxNumCars);
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
            memcpy(dev_cars_2[target], dev_cars[i], sizeof(Car));
            d_Car_active_2[target] = 1;

            // Update pointer in Cell.
            dev_cars[i]->position->car = dev_cars[target];

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

int main(int /*argc*/, char ** /*argv*/) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);

    allocate_memory();
    printf("mem alloc done\n");
    create_street_network();

    auto time_start = std::chrono::system_clock::now();

    for (int i = 0; i < kNumIterations; ++i) {
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