#include <chrono>
#include <curand_kernel.h>
#include <stdio.h>
#include "../../../mem_alloc/mem_alloc.h"
#define ALL __noinline__ __host__ __device__

#include "../configuration.h"

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

static const int kCudaBlockSize = 256;
class BodyType {

public:
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float mass;
  float force_x;
  float force_y;

public:
  __noinline__ __device__ virtual void initBody(int idx) = 0;
  ALL BodyType() {}
  ALL BodyType(int idx) {}

  ALL virtual float computeDistance(BodyType *other) = 0;
  ALL virtual float computeForce(BodyType *other, float dist) = 0;
  ALL virtual void updateVelX() = 0;
  ALL virtual void updateVelY() = 0;
  ALL virtual void updatePosX() = 0;
  ALL virtual void updatePosY() = 0;
  ALL virtual void initForce() = 0;
  ALL virtual void updateForceX(BodyType *other, float F) = 0;
  ALL virtual void updateForceY(BodyType *other, float F) = 0;

  void add_checksum() {}

  // Only for rendering.
  ALL float pos_x_() const { return pos_x; }
  ALL float pos_y_() const { return pos_y; }
  ALL float mass_() const { return mass; }
};
class Body : public BodyType {

public:
  __noinline__ __device__ void initBody(int idx) {
    curandState rand_state;
    curand_init(kSeed, idx, 0, &rand_state);

    pos_x = 2 * curand_uniform(&rand_state) - 1;
    pos_y = 2 * curand_uniform(&rand_state) - 1;
    vel_x = (curand_uniform(&rand_state) - 0.5) / 1000;
    vel_y = (curand_uniform(&rand_state) - 0.5) / 1000;
    mass = (curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass;
  }
  ALL Body() {}

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

  void add_checksum();

  // Only for rendering.
  ALL float pos_x_() const { return pos_x; }
  ALL float pos_y_() const { return pos_y; }
  ALL float mass_() const { return mass; }
};

__device__ float device_checksum;

__managed__ range_tree_node *range_tree;
__managed__ unsigned tree_size;
__managed__ void *temp_Body_compute_force;
__managed__ void *temp_update;
__global__ void Body_compute_force(BodyType **dev_bodies) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  float dist;
  float F;
  void **vtable;

  if (id < kNumBodies) {
    vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
    temp_Body_compute_force = vtable[7];
    dev_bodies[id]->initForce();
    // printf("%d ddd\n", id);
    for (int i = 0; i < kNumBodies; ++i) {
      // Do not compute force with the body itself.
      if (id != i) {
        vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
        temp_Body_compute_force = vtable[1];
        dist = dev_bodies[id]->computeDistance(dev_bodies[i]);

        vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
        temp_Body_compute_force = vtable[2];
        F = dev_bodies[id]->computeForce(dev_bodies[i], dist);

        vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
        temp_Body_compute_force = vtable[8];
        dev_bodies[id]->updateForceX(dev_bodies[i], F);

        vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
        temp_Body_compute_force = vtable[9];
        dev_bodies[id]->updateForceY(dev_bodies[i], F);
      }
    }
  }
}

__global__ void Body_update(BodyType **dev_bodies) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  void **vtable;
  if (id < kNumBodies) {

    vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
    temp_Body_compute_force = vtable[3];
    dev_bodies[id]->updateVelX();
    vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
    temp_Body_compute_force = vtable[4];
    dev_bodies[id]->updateVelY();
    vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
    temp_Body_compute_force = vtable[5];
    dev_bodies[id]->updatePosX();
    vtable = get_vfunc(dev_bodies[id], range_tree, tree_size);
    temp_Body_compute_force = vtable[6];
    dev_bodies[id]->updatePosY();

    if (dev_bodies[id]->pos_x < -1 || dev_bodies[id]->pos_x > 1) {
      dev_bodies[id]->vel_x = -dev_bodies[id]->vel_x;
    }

    if (dev_bodies[id]->pos_y < -1 || dev_bodies[id]->pos_y > 1) {
      dev_bodies[id]->vel_y = -dev_bodies[id]->vel_y;
    }
  }
  // dev_bodies[id].vel_x += dev_bodies[id].force_x * kDt / dev_bodies[id].mass;
  // dev_bodies[id].vel_y += dev_bodies[id].force_y * kDt / dev_bodies[id].mass;
  // dev_bodies[id].pos_x += dev_bodies[id].vel_x * kDt;
  // dev_bodies[id].pos_y += dev_bodies[id].vel_y * kDt;

  // if (dev_bodies[id].pos_x < -1 || dev_bodies[id].pos_x > 1) {
  //   dev_bodies[id].vel_x = -dev_bodies[id].vel_x;
  // }

  // if (dev_bodies[id].pos_y < -1 || dev_bodies[id].pos_y > 1) {
  //   dev_bodies[id].vel_y = -dev_bodies[id].vel_y;
  // }
}

__device__ void Body_add_checksum(BodyType **dev_bodies, int id) {
  atomicAdd(&device_checksum,
            dev_bodies[id]->pos_x + dev_bodies[id]->pos_y * 2 +
                dev_bodies[id]->vel_x * 3 + dev_bodies[id]->vel_y * 4);
}

void instantiate_bodies(BodyType **bodies, obj_alloc *alloc) {

  for (int i = 0; i < kNumBodies; i++)
    bodies[i] = (Body *)alloc->my_new<Body>();
}
__global__ void kernel_initialize_bodies(BodyType **bodies) {

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kNumBodies;
       i += blockDim.x * gridDim.x) {
    bodies[i]->initBody(i);
  }
}

__global__ void kernel_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kNumBodies;
       i += blockDim.x * gridDim.x) {
    // Body_compute_force(i);
  }
}

__global__ void kernel_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kNumBodies;
       i += blockDim.x * gridDim.x) {
    // Body_update(i);
  }
}

__global__ void kernel_compute_checksum(BodyType **bodies) {
  device_checksum = 0.0f;
  for (int i = 0; i < kNumBodies; ++i) {
    Body_add_checksum(bodies, i);
  }
}

int main(int /*argc*/, char ** /*argv*/) {
  BodyType **dev_bodies;
  mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
  obj_alloc my_obj_alloc(&shared_mem);
  // Allocate and create Body objects.
  dev_bodies = (BodyType **)my_obj_alloc.calloc<BodyType *>(kNumBodies);
  printf("init bodies...\n");
  instantiate_bodies(dev_bodies, &my_obj_alloc);
  my_obj_alloc.toDevice();
  kernel_initialize_bodies<<<128, 128>>>(dev_bodies);
  gpuErrchk(cudaDeviceSynchronize());
  my_obj_alloc.create_tree();
  range_tree = my_obj_alloc.get_range_tree();
  tree_size = my_obj_alloc.get_tree_size();
  printf("init done...\n");
  auto time_start = std::chrono::system_clock::now();
  printf("Kernel exec...\n");
  for (int i = 0; i < kNumIterations; ++i) {
    if (i % 300 == 0)
      printf("Start: BodyComputeForce(%d)\n", i);
    Body_compute_force<<<(kNumBodies + kCudaBlockSize - 1) / kCudaBlockSize,
                         kCudaBlockSize>>>(dev_bodies);
    gpuErrchk(cudaDeviceSynchronize());
    // printf("Body_compute_force(%d)\n",i);
    Body_update<<<(kNumBodies + kCudaBlockSize - 1) / kCudaBlockSize,
                  kCudaBlockSize>>>(dev_bodies);
    gpuErrchk(cudaDeviceSynchronize());
    if (i % 300 == 0)
      printf("Finish: BodyComputeForce(%d)\n", i);
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

  printf("%lu\n", micros);

#ifndef NDEBUG
  kernel_compute_checksum<<<1, 1>>>(dev_bodies);
  gpuErrchk(cudaDeviceSynchronize());

  float checksum;
  cudaMemcpyFromSymbol(&checksum, device_checksum, sizeof(device_checksum), 0,
                       cudaMemcpyDeviceToHost);
  printf("Checksum: %f\n", checksum);
#endif // NDEBUG

  cudaFree(dev_bodies);

  return 0;
}
