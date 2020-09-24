#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>

#include "../../../mem_alloc/mem_alloc_tp.h"
#define ALL __noinline__ __host__ __device__
#include "coal.h"
#include "../configuration.h"

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
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
        dx = this->pos_x - CLEANPTR(other,BodyType *)->pos_x;
        dy = this->pos_y - CLEANPTR(other,BodyType *)->pos_y;
        dist = sqrt(dx * dx + dy * dy);
        return dist;
    }
    ALL float computeForce(BodyType *other, float dist) {
        float F = kGravityConstant * this->mass * CLEANPTR(other,BodyType *)->mass /
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
        dx = -1 * (this->pos_x - CLEANPTR(other,BodyType *)->pos_x);
        dy = -1 * (this->pos_y - CLEANPTR(other,BodyType *)->pos_y);
        dist = sqrt(dx * dx + dy * dy);
        this->force_x += F * dx / dist;
    }
    ALL void updateForceY(BodyType *other, float F) {
        float dx;
        float dy;
        float dist;
        dx = -1 * (this->pos_x - CLEANPTR(other,BodyType *)->pos_x);
        dy = -1 * (this->pos_y - CLEANPTR(other,BodyType *)->pos_y);
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

__managed__ obj_info_tuble *vfun_table;
__managed__ unsigned tree_size_g;
__managed__ void *temp_copyBack;
__managed__ void *temp_TP;


__global__ void Body_compute_force(BodyType **dev_bodies) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    float dist;
    float F;
    void **vtable;

    if (id < kNumBodies) {
        BodyType *ptr = dev_bodies[id];
        COAL_BodyType_initForce(ptr);
        CLEANPTR(ptr,BodyType *)->initForce();
        // printf("%d ddd\n", id);
        for (int i = 0; i < kNumBodies; ++i) {
            // Do not compute force with the body itself.
            if (id != i) {
                COAL_BodyType_computeDistance(ptr);
                dist = CLEANPTR(ptr,BodyType *)->computeDistance(dev_bodies[i]);

                COAL_BodyType_computeForce(ptr);
                F = CLEANPTR(ptr,BodyType *)->computeForce(dev_bodies[i], dist);

                COAL_BodyType_updateForceX(ptr);
                CLEANPTR(ptr,BodyType *)->updateForceX(dev_bodies[i], F);

                COAL_BodyType_updateForceY(ptr);
                CLEANPTR(ptr,BodyType *)->updateForceY(dev_bodies[i], F);
            }
        }
    }
}

__global__ void Body_update(BodyType **dev_bodies) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    void **vtable;
    if (id < kNumBodies) {
        BodyType *ptr = dev_bodies[id];
       COAL_BodyType_updateVelX(ptr);
       CLEANPTR(ptr,BodyType *)->updateVelX();
        COAL_BodyType_updateVelY(ptr);
        CLEANPTR(ptr,BodyType *)->updateVelY();
        COAL_BodyType_updatePosX(ptr);
        CLEANPTR(ptr,BodyType *)->updatePosX();
        COAL_BodyType_updatePosY(ptr);
        CLEANPTR(ptr,BodyType *)->updatePosY();

        if (CLEANPTR(ptr,BodyType *)->pos_x < -1 || CLEANPTR(ptr,BodyType *)->pos_x > 1) {
            CLEANPTR(ptr,BodyType *)->vel_x = -CLEANPTR(ptr,BodyType *)->vel_x;
        }

        if (CLEANPTR(ptr,BodyType *)->pos_y < -1 || CLEANPTR(ptr,BodyType *)->pos_y > 1) {
            CLEANPTR(ptr,BodyType *)->vel_y = -CLEANPTR(ptr,BodyType *)->vel_y;
        }
    }

}

__device__ void Body_add_checksum(BodyType **dev_bodies, int id) {
    atomicAdd(&device_checksum,
        CLEANPTR(dev_bodies[id],BodyType *)->pos_x + CLEANPTR(dev_bodies[id],BodyType *)->pos_y * 2 +
        CLEANPTR(dev_bodies[id],BodyType *)->vel_x * 3 + CLEANPTR(dev_bodies[id],BodyType *)->vel_y * 4);
}

void instantiate_bodies(BodyType **bodies, obj_alloc *alloc) {
    for (int i = 0; i < kNumBodies; i++)
        bodies[i] = (Body *)alloc->my_new<Body>();
}
__global__ void kernel_initialize_bodies(BodyType **bodies) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kNumBodies;
         i += blockDim.x * gridDim.x) {
            CLEANPTR(bodies[i],BodyType *)->initBody(i);
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
    obj_alloc my_obj_alloc(&shared_mem, atoll(argv[1]));
    // Allocate and create Body objects.
    dev_bodies = (BodyType **)my_obj_alloc.calloc<BodyType *>(kNumBodies);
    printf("init bodies...\n");
    instantiate_bodies(dev_bodies, &my_obj_alloc);
    my_obj_alloc.toDevice();
    kernel_initialize_bodies<<<128, 128>>>(dev_bodies);
    gpuErrchk(cudaDeviceSynchronize());
    my_obj_alloc.create_table();
    vfun_table = my_obj_alloc.get_vfun_table();
    printf("init done...\n");
    auto time_start = std::chrono::system_clock::now();
    printf("Kernel exec...\n");
    for (int i = 0; i < kNumIterations; ++i) {
        if (i % 300 == 0) printf("Start: BodyComputeForce(%d)\n", i);
        Body_compute_force<<<(kNumBodies + kCudaBlockSize - 1) / kCudaBlockSize,
                             kCudaBlockSize>>>(dev_bodies);
        gpuErrchk(cudaDeviceSynchronize());
        // printf("Body_compute_force(%d)\n",i);
        Body_update<<<(kNumBodies + kCudaBlockSize - 1) / kCudaBlockSize,
                      kCudaBlockSize>>>(dev_bodies);
        gpuErrchk(cudaDeviceSynchronize());
        if (i % 300 == 0) printf("Finish: BodyComputeForce(%d)\n", i);
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
#endif  // NDEBUG

    cudaFree(dev_bodies);

    return 0;
}
