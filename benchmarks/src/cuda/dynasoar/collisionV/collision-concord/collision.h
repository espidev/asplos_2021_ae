#include "../configuration.h"
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>
#include <assert.h>

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif // OPTION_RENDER

#define ALL __noinline__ __host__ __device__
#define CONCORD(ptr, fun, arg)    \
    if (ptr->classType == 0)      \
        ptr->Base##fun arg;       \
    else if (ptr->classType == 1) \
        ptr->fun arg;

#define CONCORDR(r, ptr, fun, arg) \
    if (ptr->classType == 0)       \
        r = ptr->Base##fun arg;    \
    else if (ptr->classType == 1)  \
        r = ptr->fun arg;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

using IndexT = int;
static const IndexT kNullptr = std::numeric_limits<IndexT>::max();

static const int kThreads = 256;
static const int kBlocks = (kNumBodies + kThreads - 1) / kThreads;

class BodyType {
public:
  IndexT merge_target;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float force_x;
  float force_y;
  float mass;
  bool has_incoming_merge;
  bool successful_merge;
  bool is_active;
  int classType = 0;

public:
 
  ALL BodyType() { classType = 0 ;}
  __noinline__ __host__ __device__ BodyType(int idx) {classType = 0 ;}

      __noinline__ __device__ void BaseinitBody(int idx) {
    curandState rand_state;
    curand_init(kSeed, idx, 0, &rand_state);

    pos_x = 2 * curand_uniform(&rand_state) - 1;
    pos_y = 2 * curand_uniform(&rand_state) - 1;
    vel_x = (curand_uniform(&rand_state) - 0.5) / 1000;
    vel_y = (curand_uniform(&rand_state) - 0.5) / 1000;
    mass = (curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass;
     is_active = true;
     classType = 0;
  }
  ALL float BasecomputeDistance(BodyType *other) {
    float dx;
    float dy;
    float dist;
    dx = this->pos_x - other->pos_x;
    dy = this->pos_y - other->pos_y;
    dist = sqrt(dx * dx + dy * dy);
    return dist;
  }
  ALL float BasecomputeForce(BodyType *other, float dist) {
    float F = kGravityConstant * this->mass * other->mass /
              (dist * dist + kDampeningFactor);
    return F;
  }
  ALL void BaseupdateVelX() { this->vel_x += this->force_x * kDt / this->mass; }
  ALL void BaseupdateVelY() { this->vel_y += this->force_y * kDt / this->mass; }
  ALL void BaseupdatePosX() { this->pos_x += this->vel_x * kDt; }
  ALL void BaseupdatePosY() { this->pos_y += this->vel_y * kDt; }
  ALL float BaseVelX() { return this->vel_x; }
  ALL float BaseVelY() { return this->vel_y; }
  ALL float BasePosX() { return this->pos_x; }
  ALL float BasePosY() { return this->pos_y; }
  ALL void Baseset_VelX(float v) { this->vel_x = v; }
  ALL void Baseset_VelY(float v) { this->vel_y = v; }
  ALL void Baseset_PosX(float v) { this->pos_x = v; }
  ALL void Baseset_PosY(float v) { this->pos_y = v; }
  ALL void BaseinitForce() {
    this->force_x = 0;
    this->force_y = 0;
  }
  ALL void BaseupdateForceX(BodyType *other, float F) {
    float dx;
    float dy;
    float dist;
    dx = -1 * (this->pos_x - other->pos_x);
    dy = -1 * (this->pos_y - other->pos_y);
    dist = sqrt(dx * dx + dy * dy);
    this->force_x += F * dx / dist;
  }
  ALL void BaseupdateForceY(BodyType *other, float F) {
    float dx;
    float dy;
    float dist;
    dx = -1 * (this->pos_x - other->pos_x);
    dy = -1 * (this->pos_y - other->pos_y);
    dist = sqrt(dx * dx + dy * dy);
    this->force_y += F * dy / dist;
  }
  ALL bool Baseactive() { return is_active; }
  ALL void Baseset_active(bool cond) { this->is_active = cond; }
  ALL float Baseget_mass() { return this->mass; }
  ALL void Baseset_mass(float newmass) { this->mass = newmass; }
  ALL bool Baseget_incoming_merge() { return this->has_incoming_merge; }
  ALL bool Baseget_is_successful_merge() { return this->successful_merge; }
  ALL IndexT Baseget_merge_target() { return this->merge_target; }
  ALL void Baseset_incoming_merge(bool cond) { this->has_incoming_merge = cond; }
  ALL void Baseset_is_successful_merge(bool cond) { this->successful_merge = cond; }
  ALL void Baseset_merge_target(IndexT idx) { this->merge_target = idx; }

      void Baseadd_checksum();

  // BaseOnly Basefor rendering.
  ALL float Basepos_x_() const { return pos_x; }
  ALL float Basepos_y_() const { return pos_y; }
  ALL float Basemass_() const { return mass; }

  //////////////
    __noinline__ __device__ void initBody(int idx) {
    curandState rand_state;
    curand_init(kSeed, idx, 0, &rand_state);

    pos_x = 2 * curand_uniform(&rand_state) - 1;
    pos_y = 2 * curand_uniform(&rand_state) - 1;
    vel_x = (curand_uniform(&rand_state) - 0.5) / 1000;
    vel_y = (curand_uniform(&rand_state) - 0.5) / 1000;
    mass = (curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass;
     is_active = true;
     classType = 1;
  }
  ALL float computeDistance(BodyType *other) {
    float dx;
    float dy;
    float dist;
    dx = this->pos_x - other->pos_x;
    dy = this->pos_y - other->pos_y;
    dist = sqrt(dx * dx + dy * dy);
    return dist;
  }
  ALL float computeForce(BodyType *other, float dist) {
    float F = kGravityConstant * this->mass * other->mass /
              (dist * dist + kDampeningFactor);
    return F;
  }
  ALL void updateVelX() { this->vel_x += this->force_x * kDt / this->mass; }
  ALL void updateVelY() { this->vel_y += this->force_y * kDt / this->mass; }
  ALL void updatePosX() { this->pos_x += this->vel_x * kDt; }
  ALL void updatePosY() { this->pos_y += this->vel_y * kDt; }
  ALL float VelX() { return this->vel_x; }
  ALL float VelY() { return this->vel_y; }
  ALL float PosX() { return this->pos_x; }
  ALL float PosY() { return this->pos_y; }
  ALL void set_VelX(float v) { this->vel_x = v; }
  ALL void set_VelY(float v) { this->vel_y = v; }
  ALL void set_PosX(float v) { this->pos_x = v; }
  ALL void set_PosY(float v) { this->pos_y = v; }
  ALL void initForce() {
    this->force_x = 0;
    this->force_y = 0;
  }
  ALL void updateForceX(BodyType *other, float F) {
    float dx;
    float dy;
    float dist;
    dx = -1 * (this->pos_x - other->pos_x);
    dy = -1 * (this->pos_y - other->pos_y);
    dist = sqrt(dx * dx + dy * dy);
    this->force_x += F * dx / dist;
  }
  ALL void updateForceY(BodyType *other, float F) {
    float dx;
    float dy;
    float dist;
    dx = -1 * (this->pos_x - other->pos_x);
    dy = -1 * (this->pos_y - other->pos_y);
    dist = sqrt(dx * dx + dy * dy);
    this->force_y += F * dy / dist;
  }
  ALL bool active() { return is_active; }
  ALL void set_active(bool cond) { this->is_active = cond; }
  ALL float get_mass() { return this->mass; }
  ALL void set_mass(float newmass) { this->mass = newmass; }
  ALL bool get_incoming_merge() { return this->has_incoming_merge; }
  ALL bool get_is_successful_merge() { return this->successful_merge; }
  ALL IndexT get_merge_target() { return this->merge_target; }
  ALL void set_incoming_merge(bool cond) { this->has_incoming_merge = cond; }
  ALL void set_is_successful_merge(bool cond) { this->successful_merge = cond; }
  ALL void set_merge_target(IndexT idx) { this->merge_target = idx; }

  void add_checksum();

  // Only for rendering.
  ALL float pos_x_() const { return pos_x; }
  ALL float pos_y_() const { return pos_y; }
  ALL float mass_() const { return mass; }
};

class Body : public BodyType {
public:

    __noinline__ __host__ __device__ Body() {
       classType = 1;

  }
  __noinline__ __host__ __device__ Body(int idx) {
    classType = 1 ;
  }

  
};
