#include <assert.h>
#include <chrono>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <limits>

#include "../configuration.h"
#include "../../../mem_alloc_better/mem_alloc_better.h"
__managed__ vfunc_table *vfun_table;
__managed__ void *temp_coal;



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

  __device__  __noinline__ virtual float random_uni() = 0;
  __device__ __host__ __noinline__ virtual int get_current_max_velocity() = 0;

  __device__ __host__ __noinline__ virtual int get_max_velocity() = 0;

  __device__ __host__ __noinline__ virtual void set_current_max_velocity(int v) = 0;

  __device__ __host__ __noinline__ virtual void remove_speed_limit() = 0;
  __device__ __host__ __noinline__ virtual int get_num_incoming() = 0;

  __device__ __host__ __noinline__ virtual void set_num_incoming(int num) = 0;
  __device__ __host__ __noinline__ virtual int get_num_outgoing() = 0;
  __device__ __host__ __noinline__ virtual void set_num_outgoing(int num) = 0;

  __device__ __host__ __noinline__ virtual CellBase *get_incoming(int idx) = 0;
  __device__ __host__ __noinline__ virtual void set_incoming(int idx, CellBase *cell) = 0;

  __device__ __host__ __noinline__ virtual CellBase *get_outgoing(int idx) = 0;

  __device__ __host__ __noinline__ virtual void set_outgoing(int idx, CellBase *cell) = 0;

  __device__ __host__ __noinline__ virtual float get_x() = 0;

  __device__ __host__ __noinline__ virtual float get_y() = 0;

  __device__ __host__ __noinline__ virtual bool is_free() = 0;

  __device__ __host__ __noinline__ virtual bool is_sink() = 0;

  __device__ __host__ __noinline__ virtual bool get_is_target() = 0;
  __device__ __host__ __noinline__ virtual void set_target() = 0;
  __device__ __host__ __noinline__ virtual void occupy(CarBase *car) = 0;

  __device__ __host__ __noinline__ virtual void release() = 0;
};

class Cell : public CellBase {
public:
  __device__  __noinline__ float random_uni() { return curand_uniform(&this->random_state); }

  __device__ __host__ __noinline__ int get_current_max_velocity() {
    return this->current_max_velocity;
  }

  __device__ __host__ __noinline__ int get_max_velocity() { return this->max_velocity; }

  __device__ __host__ __noinline__ void set_current_max_velocity(int v) {
    this->current_max_velocity = v;
  }

  __device__ __host__ __noinline__ void remove_speed_limit() {
    this->current_max_velocity = this->max_velocity;
  }

  __device__ __host__ __noinline__ int get_num_incoming() { return this->num_incoming; }

  __device__ __host__ __noinline__ void set_num_incoming(int num) { this->num_incoming = num; }

  __device__ __host__ __noinline__ int get_num_outgoing() { return this->num_outgoing; }

  __device__ __host__ __noinline__ void set_num_outgoing(int num) { this->num_outgoing = num; }

  __device__ __host__ __noinline__ CellBase *get_incoming(int idx) { return this->incoming[idx]; }

  __device__ __host__ __noinline__ void set_incoming(int idx, CellBase *cell) {
    assert(cell != nullptr);
    this->incoming[idx] = cell;
  }

  __device__ __host__ __noinline__ CellBase *get_outgoing(int idx) { return this->outgoing[idx]; }

  __device__ __host__ __noinline__ void set_outgoing(int idx, CellBase *cell) {
    assert(cell != nullptr);
    this->outgoing[idx] = cell;
  }

  __device__ __host__ __noinline__ float get_x() { return this->x; }

  __device__ __host__ __noinline__ float get_y() { return this->y; }

  __device__ __host__ __noinline__ bool is_free() { return this->car == nullptr; }

  __device__ __host__ __noinline__ bool is_sink() { return this->num_outgoing == 0; }

  __device__ __host__ __noinline__ bool get_is_target() { return this->is_target; }

  __device__ __host__ __noinline__ void set_target() { this->is_target = true; }
  __device__ __host__ __noinline__ void occupy(CarBase *car) {
    assert(this->is_free());
    this->car = car;
  }

  __device__ __host__ __noinline__ void release() {
    assert(!this->is_free());
    this->car = nullptr;
  }
};

class CarBase {
public:
  CellBase *path[kMaxVelocity];
  CellBase *position;
  curandState_t random_state;
  int path_length;
  int velocity;
  int max_velocity;
  __device__ __host__ __noinline__ virtual void set_path(CellBase *cell, int idx) = 0;
  __device__ __host__ __noinline__ virtual CellBase *get_path(int idx) = 0;

  __device__ __host__ __noinline__ virtual void set_path_length(int len) = 0;
  __device__ __host__ __noinline__ virtual int get_path_length() = 0;
  __device__  __noinline__ virtual int random_int(int a, int b) = 0;
  __device__  __noinline__ virtual float random_uni() = 0;
  __device__ __host__ __noinline__ virtual void step_initialize_iteration() = 0;
  __device__ __host__ __noinline__ virtual int get_velocity() = 0;
  __device__ __host__ __noinline__ virtual void set_velocity(int v) = 0;
  __device__ __host__ __noinline__ virtual void set_max_velocity(int v) = 0;
  __device__ __host__ __noinline__ virtual int get_max_velocity() = 0;
  __device__ __host__ __noinline__ virtual void set_position(CellBase *cell) = 0;
  __device__ __host__ __noinline__ virtual CellBase *get_position() = 0;
  __device__  __noinline__ virtual void step_slow_down() = 0;
  __device__  __noinline__ virtual CellBase *next_step(CellBase *position) = 0;
  __device__  __noinline__ virtual void step_accelerate() = 0;
};

class Car : public CarBase {
public:
  __device__ __host__ __noinline__ void set_path(CellBase *cell, int idx) { this->path[idx] = cell; }
  __device__ __host__ __noinline__ CellBase *get_path(int idx) { return this->path[idx]; }

  __device__ __host__ __noinline__ void set_path_length(int len) { this->path_length = len; }
  __device__ __host__ __noinline__ int get_path_length() { return this->path_length; }
  __device__  __noinline__ int random_int(int a, int b) {
    return curand(&this->random_state) % (b - a) + a;
  }
  __device__  __noinline__ float random_uni() { return curand_uniform(&this->random_state); }
  __device__ __host__ __noinline__ void step_initialize_iteration() {
    // Reset calculated path. This forces cars with a random moving behavior to
    // select a new path in every iteration. Otherwise, cars might get "stuck"
    // on a full network if many cars are waiting for the one in front of them
    // in a cycle.
    this->path_length = 0;
  }
  __device__ __host__ __noinline__ int get_velocity() { return this->velocity; }
  __device__ __host__ __noinline__ void set_velocity(int v) { this->velocity = v; }
  __device__ __host__ __noinline__ void set_max_velocity(int v) { this->max_velocity = v; }

  __device__ __host__ __noinline__ int get_max_velocity() { return this->max_velocity; }
  __device__ __host__ __noinline__ void set_position(CellBase *cell) { this->position = cell; }

  __device__ __host__ __noinline__ CellBase *get_position() { return this->position; }
  __device__  __noinline__ void step_slow_down() {
    // 20% change of slowdown.
    if (curand_uniform(&this->random_state) < 0.2 && this->velocity > 0) {
      this->velocity = this->velocity - 1;
    }
  }
  __device__  __noinline__ CellBase *next_step(CellBase *position) {
    // Almost random walk.
    const uint32_t num_outgoing = position->num_outgoing;
    assert(num_outgoing > 0);

    // Need some kind of return statement here.
    return position->outgoing[this->random_int(0, num_outgoing)];
  }
  __device__  __noinline__ void step_accelerate() {
    // Speed up the car by 1 or 2 units.

    int speedup = curand(&this->random_state) % (2 - 0) + 0 + 1;
    this->velocity = this->max_velocity < this->velocity + speedup
                         ? this->max_velocity
                         : this->velocity + speedup;
  }
};

class TrafficLightBase {
protected:
  CellBase *cells_[kMaxDegree];
  int num_cells;
  int timer;
  int phase_time;
  int phase;

public:
  __device__ __host__ __noinline__ TrafficLightBase() {}
  __device__ __host__ __noinline__ TrafficLightBase(int num_cells_, int phase_time_)
      : num_cells(num_cells_), timer(0), phase_time(phase_time_), phase(0) {}

  __device__ __host__ __noinline__ virtual void set_cell(int idx, CellBase *cell) = 0;
  __device__ __host__ __noinline__ virtual CellBase *get_cell(int idx) = 0;
  __device__ __host__ __noinline__ virtual int get_num_cells() = 0;
  __device__ __host__ __noinline__ virtual int get_timer() = 0;
  __device__ __host__ __noinline__ virtual int set_timer(int time) = 0;
  __device__ __host__ __noinline__ virtual int get_phase_time() = 0;
  __device__ __host__ __noinline__ virtual void set_phase(int ph) = 0;
  __device__ __host__ __noinline__ virtual int get_phase() = 0;

};

class TrafficLight : public TrafficLightBase {
private:
public:
  __device__ __host__ __noinline__ TrafficLight() {}
  __device__ __host__ __noinline__ TrafficLight(int num_cells_, int phase_time_) {

    num_cells = (num_cells_);
    timer = (0);
    phase_time = (phase_time_);
    phase = (0);
  }

  __device__ __host__ __noinline__ void set_cell(int idx, CellBase *cell) {
    assert(cell != nullptr);
    cells_[idx] = cell;
  }
  __device__ __host__ __noinline__ CellBase *get_cell(int idx) { return cells_[idx]; }
  __device__ __host__ __noinline__ int get_num_cells() { return num_cells; }
  __device__ __host__ __noinline__ int get_timer() { return timer; }
  __device__ __host__ __noinline__ int set_timer(int time) { return this->timer = time; }
  __device__ __host__ __noinline__ int get_phase_time() { return phase_time; }
  __device__ __host__ __noinline__ void set_phase(int ph) { this->phase = ph; }
  __device__ __host__ __noinline__ int get_phase() { return this->phase; }
 
};





class dummyX {
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

  __device__  __noinline__ virtual float random_uni() = 0;
  __device__ __host__ __noinline__ virtual int get_current_max_velocity() = 0;

  __device__ __host__ __noinline__ virtual int get_max_velocity() = 0;

  __device__ __host__ __noinline__ virtual void set_current_max_velocity(int v) = 0;

  __device__ __host__ __noinline__ virtual void remove_speed_limit() = 0;
  __device__ __host__ __noinline__ virtual int get_num_incoming() = 0;

  __device__ __host__ __noinline__ virtual void set_num_incoming(int num) = 0;
  __device__ __host__ __noinline__ virtual int get_num_outgoing() = 0;
  __device__ __host__ __noinline__ virtual void set_num_outgoing(int num) = 0;

  __device__ __host__ __noinline__ virtual CellBase *get_incoming(int idx) = 0;
  __device__ __host__ __noinline__ virtual void set_incoming(int idx, CellBase *cell) = 0;

  __device__ __host__ __noinline__ virtual CellBase *get_outgoing(int idx) = 0;

  __device__ __host__ __noinline__ virtual void set_outgoing(int idx, CellBase *cell) = 0;

  __device__ __host__ __noinline__ virtual float get_x() = 0;

  __device__ __host__ __noinline__ virtual float get_y() = 0;

  __device__ __host__ __noinline__ virtual bool is_free() = 0;

  __device__ __host__ __noinline__ virtual bool is_sink() = 0;

  __device__ __host__ __noinline__ virtual bool get_is_target() = 0;
  __device__ __host__ __noinline__ virtual void set_target() = 0;
  __device__ __host__ __noinline__ virtual void occupy(CarBase *car) = 0;

  __device__ __host__ __noinline__ virtual void release() = 0;
};


class dummy : public dummyX{

public:
    __device__  __noinline__ float random_uni() { return curand_uniform(&this->random_state); }
  
    __device__ __host__ __noinline__ int get_current_max_velocity() {
      return this->current_max_velocity;
    }
  
    __device__ __host__ __noinline__ int get_max_velocity() { return this->max_velocity; }
  
    __device__ __host__ __noinline__ void set_current_max_velocity(int v) {
      this->current_max_velocity = v;
    }
  
    __device__ __host__ __noinline__ void remove_speed_limit() {
      this->current_max_velocity = this->max_velocity;
    }
  
    __device__ __host__ __noinline__ int get_num_incoming() { return this->num_incoming; }
  
    __device__ __host__ __noinline__ void set_num_incoming(int num) { this->num_incoming = num; }
  
    __device__ __host__ __noinline__ int get_num_outgoing() { return this->num_outgoing; }
  
    __device__ __host__ __noinline__ void set_num_outgoing(int num) { this->num_outgoing = num; }
  
    __device__ __host__ __noinline__ CellBase *get_incoming(int idx) { return this->incoming[idx]; }
  
    __device__ __host__ __noinline__ void set_incoming(int idx, CellBase *cell) {
      assert(cell != nullptr);
      this->incoming[idx] = cell;
    }
  
    __device__ __host__ __noinline__ CellBase *get_outgoing(int idx) { return this->outgoing[idx]; }
  
    __device__ __host__ __noinline__ void set_outgoing(int idx, CellBase *cell) {
      assert(cell != nullptr);
      this->outgoing[idx] = cell;
    }
  
    __device__ __host__ __noinline__ float get_x() { return this->x; }
  
    __device__ __host__ __noinline__ float get_y() { return this->y; }
  
    __device__ __host__ __noinline__ bool is_free() { return this->car == nullptr; }
  
    __device__ __host__ __noinline__ bool is_sink() { return this->num_outgoing == 0; }
  
    __device__ __host__ __noinline__ bool get_is_target() { return this->is_target; }
  
    __device__ __host__ __noinline__ void set_target() { this->is_target = true; }
    __device__ __host__ __noinline__ void occupy(CarBase *car) {
      assert(this->is_free());
      this->car = car;
    }
  
    __device__ __host__ __noinline__ void release() {
      assert(!this->is_free());
      this->car = nullptr;
    }
};