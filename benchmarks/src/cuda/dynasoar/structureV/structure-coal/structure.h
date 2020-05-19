#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>
#include "coal.h"
#include "../configuration.h"
#include "../dataset.h"
#include "../../../mem_alloc/mem_alloc.h"
__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_coal;
class SpringBase;
class NodeBase {
public:
  SpringBase *springs[kMaxDegree];
  int num_springs;
  int distance;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float mass;
  char type;
  __device__ virtual SpringBase *spring(unsigned i) = 0;
  __device__ virtual void pull() = 0;
  __device__ virtual float distance_to(NodeBase *other) = 0;
  __device__ virtual void remove_spring(SpringBase *spring) = 0;
  __device__ virtual float unit_x(NodeBase *other, float dist) = 0;
  __device__ virtual float unit_y(NodeBase *other, float dist) = 0;
  __device__ virtual void update_vel_x(float force_x) = 0;
  __device__ virtual void update_vel_y(float force_y) = 0;
  __device__ virtual void update_pos_x(float force_x) = 0;
  __device__ virtual void update_pos_y(float force_x) = 0;
  __device__ virtual void set_distance(float dist) = 0;
  __device__ virtual float get_distance() = 0;
};

class Node : public NodeBase {
public:
  __device__ SpringBase *spring(unsigned i) { return this->springs[i]; }
  __device__ void pull() {
    this->pos_x += this->vel_x * kDt;
    this->pos_y += this->vel_y * kDt;
  }
  __device__ float distance_to(NodeBase *other) {
    float dx = this->pos_x - other->pos_x;
    float dy = this->pos_y - other->pos_y;
    float dist_sq = dx * dx + dy * dy;
    return sqrt(dist_sq);
  }
  __device__ void remove_spring(SpringBase *spring) {

    for (int i = 0; i < kMaxDegree; ++i) {
      if (this->springs[i] == spring) {
        this->springs[i] = NULL;
        if (atomicSub(&this->num_springs, 1) == 1) {
          // Deleted last spring.
          this->type = 0;
        }
        return;
      }
    }

    // Spring not found.
    assert(false);
  }
  __device__ float unit_x(NodeBase *other, float dist) {
    float dx = this->pos_x - other->pos_x;
    float unit_x = dx / dist;
    return unit_x;
  }
  __device__ float unit_y(NodeBase *other, float dist) {
    float dy = this->pos_y - other->pos_y;
    float unit_y = dy / dist;
    return unit_y;
  }
  __device__ void update_vel_x(float force_x) {
    this->vel_x += force_x * kDt / this->mass;
    this->vel_x *= 1.0f - kVelocityDampening;
  }
  __device__ void update_vel_y(float force_y) {
    this->vel_y += force_y * kDt / this->mass;
    this->vel_y *= 1.0f - kVelocityDampening;
  }
  __device__ void update_pos_x(float force_x) {
    this->pos_x += this->vel_x * kDt;
  }
  __device__ void update_pos_y(float force_x) {
    this->pos_y += this->vel_y * kDt;
  }
  __device__ void set_distance(float dist) { this->distance = dist; }
  __device__ float get_distance() { return this->distance; }
};

class SpringBase {
public:
  float factor;
  float initial_length;
  float force;
  float max_force;
  bool is_active;

  NodeBase *p1;
  NodeBase *p2;
  bool delete_flag;
  __device__ virtual void deactivate() =0 ;
  __device__ virtual void update_force(float displacement) =0 ;
  __device__ virtual float get_force() = 0;
  __device__ virtual float get_init_len()  = 0;
  __device__ virtual bool get_is_active()  = 0;
  __device__ virtual void set_is_active(bool active)  = 0;
  __device__ virtual NodeBase *get_p1()  = 0;
  __device__ virtual NodeBase *get_p2()  = 0;
  __device__ virtual bool is_max_force()  = 0;
};

class Spring: public SpringBase {
public:

  __device__ void deactivate() { this->is_active = false; }
  __device__ void update_force(float displacement) {
    this->force = this->factor * displacement;
  }
  __device__ float get_force() { return this->force; }
  __device__ float get_init_len() { return this->initial_length; }
  __device__ bool get_is_active() { return this->is_active; }
  __device__ void set_is_active(bool active) { this->is_active = active; }
  __device__ NodeBase *get_p1() { return this->p1; }
  __device__ NodeBase *get_p2() { return this->p2; }
  __device__ bool is_max_force() { return this->force > this->max_force; }
};
