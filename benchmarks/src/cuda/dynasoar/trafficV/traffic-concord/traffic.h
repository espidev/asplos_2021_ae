#include <assert.h>
#include <curand_kernel.h>
#include <chrono>
#include <cub/cub.cuh>
#include <limits>
#define CONCORD2(ptr, fun)        \
    if (ptr->classType == 0)      \
        ptr->Base##fun;           \
    else if (ptr->classType == 1) \
        ptr->fun;
#define CONCORD3(r, ptr, fun)     \
    if (ptr->classType == 0)      \
        r = ptr->Base##fun;       \
    else if (ptr->classType == 1) \
        r = ptr->fun;

#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define CONCORD(...) \
    GET_MACRO(__VA_ARGS__, CONCORD4, CONCORD3, CONCORD2, CONCORD1)(__VA_ARGS__)

#include "../configuration.h"
//#include "util/util.h"
class CarBase;
class TrafficLightBase;
class CellBase {
  public:
    curandState_t random_state;
    CellBase *incoming[kMaxDegree];
    CellBase *outgoing[kMaxDegree];
    CarBase *car;
    int num_incoming;
    int num_outgoing;
    int max_velocity;
    int current_max_velocity;
    float x;
    float y;
    bool is_target;
    char type;
    int classType = 0;

    __noinline__ __host__ __device__ CellBase() { classType = 0; }
     __noinline__ __device__ float Baserandom_uni() {
        return curand_uniform(&this->random_state);
    }

    __noinline__ __host__ __device__ int Baseget_current_max_velocity() {
        return this->current_max_velocity;
    }

    __noinline__ __host__ __device__ int Baseget_max_velocity() {
        return this->max_velocity;
    }

    __noinline__ __host__ __device__ void Baseset_current_max_velocity(int v) {
        this->current_max_velocity = v;
    }

    __noinline__ __host__ __device__ void Baseremove_speed_limit() {
        this->current_max_velocity = this->max_velocity;
    }

    __noinline__ __host__ __device__ int Baseget_num_incoming() {
        return this->num_incoming;
    }

    __noinline__ __host__ __device__ void Baseset_num_incoming(int num) {
        this->num_incoming = num;
    }

    __noinline__ __host__ __device__ int Baseget_num_outgoing() {
        return this->num_outgoing;
    }

    __noinline__ __host__ __device__ void Baseset_num_outgoing(int num) {
        this->num_outgoing = num;
    }

    __noinline__ __host__ __device__ CellBase *Baseget_incoming(int idx) {
        return this->incoming[idx];
    }

    __noinline__ __host__ __device__ void Baseset_incoming(int idx,
                                                       CellBase *cell) {
        assert(cell != nullptr);
        this->incoming[idx] = cell;
    }

    __noinline__ __host__ __device__ CellBase *Baseget_outgoing(int idx) {
        return this->outgoing[idx];
    }

    __noinline__ __host__ __device__ void Baseset_outgoing(int idx,
                                                       CellBase *cell) {
        assert(cell != nullptr);
        this->outgoing[idx] = cell;
    }

    __noinline__ __host__ __device__ float Baseget_x() { return this->x; }

    __noinline__ __host__ __device__ float Baseget_y() { return this->y; }

    __noinline__ __host__ __device__ bool Baseis_free() {
        return this->car == nullptr;
    }

    __noinline__ __host__ __device__ bool Baseis_sink() {
        return this->num_outgoing == 0;
    }

    __noinline__ __host__ __device__ bool Baseget_is_target() {
        return this->is_target;
    }

    __noinline__ __host__ __device__ void Baseset_target() {
        this->is_target = true;
    }
    __noinline__ __host__ __device__ void Baseoccupy(CarBase *car) {
        assert(this->is_free());
        this->car = car;
    }

    __noinline__ __host__ __device__ void Baserelease() {
        assert(!this->is_free());
        this->car = nullptr;
    }


    ///////////////

    __noinline__ __device__ float random_uni() {
        return curand_uniform(&this->random_state);
    }

    __noinline__ __host__ __device__ int get_current_max_velocity() {
        return this->current_max_velocity;
    }

    __noinline__ __host__ __device__ int get_max_velocity() {
        return this->max_velocity;
    }

    __noinline__ __host__ __device__ void set_current_max_velocity(int v) {
        this->current_max_velocity = v;
    }

    __noinline__ __host__ __device__ void remove_speed_limit() {
        this->current_max_velocity = this->max_velocity;
    }

    __noinline__ __host__ __device__ int get_num_incoming() {
        return this->num_incoming;
    }

    __noinline__ __host__ __device__ void set_num_incoming(int num) {
        this->num_incoming = num;
    }

    __noinline__ __host__ __device__ int get_num_outgoing() {
        return this->num_outgoing;
    }

    __noinline__ __host__ __device__ void set_num_outgoing(int num) {
        this->num_outgoing = num;
    }

    __noinline__ __host__ __device__ CellBase *get_incoming(int idx) {
        return this->incoming[idx];
    }

    __noinline__ __host__ __device__ void set_incoming(int idx,
                                                       CellBase *cell) {
        assert(cell != nullptr);
        this->incoming[idx] = cell;
    }

    __noinline__ __host__ __device__ CellBase *get_outgoing(int idx) {
        return this->outgoing[idx];
    }

    __noinline__ __host__ __device__ void set_outgoing(int idx,
                                                       CellBase *cell) {
        assert(cell != nullptr);
        this->outgoing[idx] = cell;
    }

    __noinline__ __host__ __device__ float get_x() { return this->x; }

    __noinline__ __host__ __device__ float get_y() { return this->y; }

    __noinline__ __host__ __device__ bool is_free() {
        return this->car == nullptr;
    }

    __noinline__ __host__ __device__ bool is_sink() {
        return this->num_outgoing == 0;
    }

    __noinline__ __host__ __device__ bool get_is_target() {
        return this->is_target;
    }

    __noinline__ __host__ __device__ void set_target() {
        this->is_target = true;
    }
    __noinline__ __host__ __device__ void occupy(CarBase *car) {
        assert(this->is_free());
        this->car = car;
    }

    __noinline__ __host__ __device__ void release() {
        assert(!this->is_free());
        this->car = nullptr;
    }
};

class Cell : public CellBase {
  public:
    __noinline__ __host__ __device__ Cell() { classType = 1; }
};

class CarBase {
  public:
    CellBase *path[kMaxVelocity];
    CellBase *position;
    curandState_t random_state;
    int path_length;
    int velocity;
    int max_velocity;
    int classType = 0;
    __noinline__ __host__ __device__ CarBase(){ classType = 0;}
     __noinline__ __host__ __device__ void Baseset_path(CellBase *cell, int idx) {
        this->path[idx] = cell;
    }
    __noinline__ __host__ __device__ CellBase *Baseget_path(int idx) {
        return this->path[idx];
    }

    __noinline__ __host__ __device__ void Baseset_path_length(int len) {
        this->path_length = len;
    }
    __noinline__ __host__ __device__ int Baseget_path_length() {
        return this->path_length;
    }
    __noinline__ __device__ int Baserandom_int(int a, int b) {
        return curand(&this->random_state) % (b - a) + a;
    }
    __noinline__ __device__ float Baserandom_uni() {
        return curand_uniform(&this->random_state);
    }
    __noinline__ __host__ __device__ void Basestep_initialize_iteration() {
        // Reset calculated path. This forces cars with a random moving behavior
        // to select a new path in every iteration. Otherwise, cars might get
        // "stuck" on a full network if many cars are waiting for the one in
        // front of them in a cycle.
        this->path_length = 0;
    }
    __noinline__ __host__ __device__ int Baseget_velocity() {
        return this->velocity;
    }
    __noinline__ __host__ __device__ void Baseset_velocity(int v) {
        this->velocity = v;
    }
    __noinline__ __host__ __device__ void Baseset_max_velocity(int v) {
        this->max_velocity = v;
    }

    __noinline__ __host__ __device__ int Baseget_max_velocity() {
        return this->max_velocity;
    }
    __noinline__ __host__ __device__ void Baseset_position(CellBase *cell) {
        this->position = cell;
    }

    __noinline__ __host__ __device__ CellBase *Baseget_position() {
        return this->position;
    }
    __noinline__ __device__ void Basestep_slow_down() {
        // 20% change of slowdown.
        if (curand_uniform(&this->random_state) < 0.2 && this->velocity > 0) {
            this->velocity = this->velocity - 1;
        }
    }
    __noinline__ __device__ CellBase *Basenext_step(CellBase *position) {
        // Almost random walk.
        const uint32_t num_outgoing = position->num_outgoing;
        assert(num_outgoing > 0);

        // Need some kind of return statement here.
        return position->outgoing[this->random_int(0, num_outgoing)];
    }
    __noinline__ __device__ void Basestep_accelerate() {
        // Speed up the car by 1 or 2 units.

        int speedup = curand(&this->random_state) % (2 - 0) + 0 + 1;
        this->velocity = this->max_velocity < this->velocity + speedup
                             ? this->max_velocity
                             : this->velocity + speedup;
    }

    ////////////////////////

    __noinline__ __host__ __device__ void set_path(CellBase *cell, int idx) {
        this->path[idx] = cell;
    }
    __noinline__ __host__ __device__ CellBase *get_path(int idx) {
        return this->path[idx];
    }

    __noinline__ __host__ __device__ void set_path_length(int len) {
        this->path_length = len;
    }
    __noinline__ __host__ __device__ int get_path_length() {
        return this->path_length;
    }
    __noinline__ __device__ int random_int(int a, int b) {
        return curand(&this->random_state) % (b - a) + a;
    }
    __noinline__ __device__ float random_uni() {
        return curand_uniform(&this->random_state);
    }
    __noinline__ __host__ __device__ void step_initialize_iteration() {
        // Reset calculated path. This forces cars with a random moving behavior
        // to select a new path in every iteration. Otherwise, cars might get
        // "stuck" on a full network if many cars are waiting for the one in
        // front of them in a cycle.
        this->path_length = 0;
    }
    __noinline__ __host__ __device__ int get_velocity() {
        return this->velocity;
    }
    __noinline__ __host__ __device__ void set_velocity(int v) {
        this->velocity = v;
    }
    __noinline__ __host__ __device__ void set_max_velocity(int v) {
        this->max_velocity = v;
    }

    __noinline__ __host__ __device__ int get_max_velocity() {
        return this->max_velocity;
    }
    __noinline__ __host__ __device__ void set_position(CellBase *cell) {
        this->position = cell;
    }

    __noinline__ __host__ __device__ CellBase *get_position() {
        return this->position;
    }
    __noinline__ __device__ void step_slow_down() {
        // 20% change of slowdown.
        if (curand_uniform(&this->random_state) < 0.2 && this->velocity > 0) {
            this->velocity = this->velocity - 1;
        }
    }
    __noinline__ __device__ CellBase *next_step(CellBase *position) {
        // Almost random walk.
        const uint32_t num_outgoing = position->num_outgoing;
        assert(num_outgoing > 0);

        // Need some kind of return statement here.
        return position->outgoing[this->random_int(0, num_outgoing)];
    }
    __noinline__ __device__ void step_accelerate() {
        // Speed up the car by 1 or 2 units.

        int speedup = curand(&this->random_state) % (2 - 0) + 0 + 1;
        this->velocity = this->max_velocity < this->velocity + speedup
                             ? this->max_velocity
                             : this->velocity + speedup;
    }
};

class Car : public CarBase {
  public:
  __noinline__ __host__ __device__ Car(){ classType = 1;}
};

class TrafficLightBase {
  protected:
    CellBase *cells_[kMaxDegree];
    int num_cells;
    int timer;
    int phase_time;
    int phase;
    

  public:
  int classType = 0;
    __noinline__ __host__ __device__ TrafficLightBase() { classType = 0 ;}
    __noinline__ __host__ __device__ TrafficLightBase(int num_cells_,
                                                      int phase_time_)
        : num_cells(num_cells_), timer(0), phase_time(phase_time_), phase(0) { classType = 0 ;}

  __noinline__ __host__ __device__ void Baseset_cell(int idx, CellBase *cell) {
        assert(cell != nullptr);
        cells_[idx] = cell;
    }
    __noinline__ __host__ __device__ CellBase *Baseget_cell(int idx) {
        return cells_[idx];
    }
    __noinline__ __host__ __device__ int Baseget_num_cells() { return num_cells; }
    __noinline__ __host__ __device__ int Baseget_timer() { return timer; }
    __noinline__ __host__ __device__ int Baseset_timer(int time) {
        return this->timer = time;
    }
    __noinline__ __host__ __device__ int Baseget_phase_time() { return phase_time; }
    __noinline__ __host__ __device__ void Baseset_phase(int ph) {
        this->phase = ph;
    }
    __noinline__ __host__ __device__ int Baseget_phase() { return this->phase; }

    __noinline__ __host__ __device__ void set_cell(int idx, CellBase *cell) {
        assert(cell != nullptr);
        cells_[idx] = cell;
    }
    __noinline__ __host__ __device__ CellBase *get_cell(int idx) {
        return cells_[idx];
    }
    __noinline__ __host__ __device__ int get_num_cells() { return num_cells; }
    __noinline__ __host__ __device__ int get_timer() { return timer; }
    __noinline__ __host__ __device__ int set_timer(int time) {
        return this->timer = time;
    }
    __noinline__ __host__ __device__ int get_phase_time() { return phase_time; }
    __noinline__ __host__ __device__ void set_phase(int ph) {
        this->phase = ph;
    }
    __noinline__ __host__ __device__ int get_phase() { return this->phase; }
};

class TrafficLight : public TrafficLightBase {
  private:
  public:
    __noinline__ __host__ __device__ TrafficLight() {classType = 1 ;}
    __noinline__ __host__ __device__ TrafficLight(int num_cells_,
                                                  int phase_time_) {
        num_cells = (num_cells_);
        timer = (0);
        phase_time = (phase_time_);
        phase = (0);
        classType = 1 ;
    }
};
