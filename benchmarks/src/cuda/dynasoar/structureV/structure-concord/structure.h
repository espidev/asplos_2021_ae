#include <assert.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>
#include <limits>
#define ALL __noinline__ __host__ __device__
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
#include "../dataset.h"
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
    int classType = 0;
    ALL NodeBase() { classType = 0; }
    ALL SpringBase *Basespring(unsigned i) { return nullptr; }
    ALL void Basepull() { return; }
    ALL float Basedistance_to(NodeBase *other) { return 0; }
    __device__ __noinline__ void Baseremove_spring(SpringBase *spring) {
        return;
    }
    ALL float Baseunit_x(NodeBase *other, float dist) { return 0; }
    ALL float Baseunit_y(NodeBase *other, float dist) { return 0; }
    ALL void Baseupdate_vel_x(float force_x) { return; }
    ALL void Baseupdate_vel_y(float force_y) { return; }
    ALL void Baseupdate_pos_x(float force_x) { return; }
    ALL void Baseupdate_pos_y(float force_x) { return; }
    ALL void Baseset_distance(float dist) { return; }
    ALL float Baseget_distance() { return 0; }

    //////

    ALL SpringBase *spring(unsigned i) { return this->springs[i]; }
    ALL void pull() {
        this->pos_x += this->vel_x * kDt;
        this->pos_y += this->vel_y * kDt;
    }
    ALL float distance_to(NodeBase *other) {
        float dx = this->pos_x - other->pos_x;
        float dy = this->pos_y - other->pos_y;
        float dist_sq = dx * dx + dy * dy;
        return sqrt(dist_sq);
    }
    __device__ __noinline__ void remove_spring(SpringBase *spring) {
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
    ALL float unit_x(NodeBase *other, float dist) {
        float dx = this->pos_x - other->pos_x;
        float unit_x = dx / dist;
        return unit_x;
    }
    ALL float unit_y(NodeBase *other, float dist) {
        float dy = this->pos_y - other->pos_y;
        float unit_y = dy / dist;
        return unit_y;
    }
    ALL void update_vel_x(float force_x) {
        this->vel_x += force_x * kDt / this->mass;
        this->vel_x *= 1.0f - kVelocityDampening;
    }
    ALL void update_vel_y(float force_y) {
        this->vel_y += force_y * kDt / this->mass;
        this->vel_y *= 1.0f - kVelocityDampening;
    }
    ALL void update_pos_x(float force_x) { this->pos_x += this->vel_x * kDt; }
    ALL void update_pos_y(float force_x) { this->pos_y += this->vel_y * kDt; }
    ALL void set_distance(float dist) { this->distance = dist; }
    ALL float get_distance() { return this->distance; }
};

class Node : public NodeBase {
  public:
    ALL Node() { classType = 1; }
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
    int classType = 0;
    ALL SpringBase(){classType = 0;}
    ALL void Basedeactivate() {}
    ALL void Baseupdate_force(float displacement) {}
    ALL float Baseget_force() { return 0; }
    ALL float Baseget_init_len() { return 0; }
    ALL bool Baseget_is_active() { return false; }
    ALL void Baseset_is_active(bool active) { return; }
    ALL NodeBase *Baseget_p1() { return nullptr; }
    ALL NodeBase *Baseget_p2() { return nullptr; }
    ALL bool Baseis_max_force() { return false; }

    /////

    ALL void deactivate() { this->is_active = false; }
    ALL void update_force(float displacement) {
        this->force = this->factor * displacement;
    }
    ALL float get_force() { return this->force; }
    ALL float get_init_len() { return this->initial_length; }
    ALL bool get_is_active() { return this->is_active; }
    ALL void set_is_active(bool active) { this->is_active = active; }
    ALL NodeBase *get_p1() { return this->p1; }
    ALL NodeBase *get_p2() { return this->p2; }
    ALL bool is_max_force() { return this->force > this->max_force; }
};

class Spring : public SpringBase {
  public:
   ALL Spring(){classType = 1;}
};
