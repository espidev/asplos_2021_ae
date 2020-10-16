
#include "collision.h"
__managed__ BodyType **d_bodies;

// Helper variables for rendering and checksum computation.
__device__ int r_draw_counter = 0;
__device__ float r_Body_pos_x[kNumBodies];
__device__ float r_Body_pos_y[kNumBodies];
__device__ float r_Body_vel_x[kNumBodies];
__device__ float r_Body_vel_y[kNumBodies];
__device__ float r_Body_mass[kNumBodies];
int host_draw_counter;
float host_Body_pos_x[kNumBodies];
float host_Body_pos_y[kNumBodies];
float host_Body_vel_x[kNumBodies];
float host_Body_vel_y[kNumBodies];
float host_Body_mass[kNumBodies];
float host_Body_is_active[kNumBodies];

__device__ void Body_apply_force(IndexT id, IndexT other) {
    void **vtable;
    if (other != id) {
        BodyType *ptr = d_bodies[other];
        COAL_BodyType_computeDistance(ptr);
        float dist =     CLEANPTR(ptr,BodyType *)->computeDistance(d_bodies[id]);
        COAL_BodyType_computeForce(ptr);
        float F =  CLEANPTR(ptr,BodyType *)->computeForce(d_bodies[id], dist);
        COAL_BodyType_updateForceX(ptr);
        CLEANPTR(ptr,BodyType *)->updateForceX(d_bodies[id], F);
        COAL_BodyType_updateForceY(ptr);
        CLEANPTR(ptr,BodyType *)->updateForceY(d_bodies[id], F);
    }
}

__device__ void Body_compute_force(IndexT id) {
    void **vtable;
    COAL_BodyType_initForce(d_bodies[id]);
    CLEANPTR(d_bodies[id],BodyType *)->initForce();

    // device_do
    for (IndexT i = 0; i < kNumBodies; ++i) {
        COAL_BodyType_active(d_bodies[i]);
        if (    CLEANPTR(d_bodies[i],BodyType *)->active()) {
            Body_apply_force(i, id);
        }
    }
}

__device__ void Body_update(IndexT id) {
    void **vtable;
    BodyType *ptr = d_bodies[id];
    COAL_BodyType_updateVelX(ptr);
    CLEANPTR(ptr,BodyType *)->updateVelX();
    COAL_BodyType_updateVelY(ptr);
    CLEANPTR(ptr,BodyType *)->updateVelY();
    COAL_BodyType_updatePosX(ptr);
    CLEANPTR(ptr,BodyType *)->updatePosX();
    COAL_BodyType_updatePosX(ptr);
    CLEANPTR(ptr,BodyType *)->updatePosY();
    COAL_BodyType_PosX(ptr);
    float idposx = CLEANPTR(ptr,BodyType *)->PosX();
    COAL_BodyType_PosY(ptr);
    float idposy = CLEANPTR(ptr,BodyType *)->PosY();
    COAL_BodyType_VelX(ptr);
    float idvelx = CLEANPTR(ptr,BodyType *)->VelX();
    COAL_BodyType_VelY(ptr);
    float idvely = CLEANPTR(ptr,BodyType *)->VelY();

    if (idposx < -1 || idposx > 1) {
        COAL_BodyType_set_VelX(ptr);
        CLEANPTR(ptr,BodyType *)->set_VelX(-idvelx);
    }

    if (idposy < -1 || idposy > 1) {
        COAL_BodyType_set_VelY(ptr);
        CLEANPTR(ptr,BodyType *)->set_VelY(-idvely);
    }
}
// __device__ void Body_update(IndexT id) {
//   d_bodies[id]->vel_x += d_bodies[id]->force_x*kTimeInterval /
//   d_bodies[id]->mass; d_bodies[id]->vel_y +=
//   d_bodies[id]->force_y*kTimeInterval / d_bodies[id]->mass;
//   d_bodies[id]->pos_x += d_bodies[id]->vel_x*kTimeInterval;
//   d_bodies[id]->pos_y += d_bodies[id]->vel_y*kTimeInterval;

//   if (d_bodies[id]->pos_x < -1 || d_bodies[id]->pos_x > 1) {
//     d_bodies[id]->vel_x = -d_bodies[id]->vel_x;
//   }

//   if (d_bodies[id]->pos_y < -1 || d_bodies[id]->pos_y > 1) {
//     d_bodies[id]->vel_y = -d_bodies[id]->vel_y;
//   }
// }
__device__ void Body_check_merge_into_this(IndexT id, IndexT other) {
    // Only merge into larger body.
    void **vtable;
    BodyType *otherPtr = d_bodies[other];
    COAL_BodyType_get_incoming_merge(otherPtr);
    bool cond1 = CLEANPTR(otherPtr,BodyType *)->get_incoming_merge();
    COAL_BodyType_get_mass(otherPtr);
    float othermass = CLEANPTR(otherPtr,BodyType *)->get_mass();
    COAL_BodyType_get_mass(otherPtr);
    float idmass = CLEANPTR(d_bodies[id],BodyType *)->get_mass();

    if (!cond1 && idmass > othermass) {
        COAL_BodyType_computeDistance(d_bodies[id]);
        float dist_square = CLEANPTR(d_bodies[id],BodyType *)->computeDistance(otherPtr);
        dist_square *= dist_square;
        if (dist_square < kMergeThreshold * kMergeThreshold) {
            // Try to merge this one.
            // There is a race condition here: Multiple threads may try to merge
            // this body.
            COAL_BodyType_set_merge_target(d_bodies[id]);
            CLEANPTR(d_bodies[id],BodyType *)->set_merge_target(other);
            COAL_BodyType_set_incoming_merge(otherPtr);
            CLEANPTR(otherPtr,BodyType *)->set_incoming_merge(true);
        }
    }
}

__device__ void Body_initialize_merge(IndexT id) {
    void **vtable;
    BodyType *ptr = d_bodies[id];
    COAL_BodyType_set_merge_target(ptr);
    CLEANPTR(ptr,BodyType *)->set_merge_target(kNullptr);
    COAL_BodyType_set_incoming_merge(ptr);
    CLEANPTR(ptr,BodyType *)->set_incoming_merge(false);
    COAL_BodyType_set_is_successful_merge(ptr);
    CLEANPTR(ptr,BodyType *)->set_is_successful_merge(false);
}

__device__ void Body_prepare_merge(IndexT id) {
    // device_do
    void **vtable;
    for (IndexT i = 0; i < kNumBodies; ++i) {
        COAL_BodyType_active(d_bodies[i]);
        if (CLEANPTR(d_bodies[i],BodyType *)->active()) {
            Body_check_merge_into_this(i, id);
        }
    }
}

// __device__ void Body_update_merge(IndexT id) {
//   IndexT m = d_bodies[id]->merge_target;
//   if (m != kNullptr) {
//     if (d_bodies[m]->merge_target == kNullptr) {
//       // Perform merge.
//       float new_mass = d_bodies[id]->mass + d_bodies[m]->mass;
//       float new_vel_x = (d_bodies[id]->vel_x*d_bodies[id]->mass
//                          + d_bodies[m]->vel_x*d_bodies[m]->mass) / new_mass;
//       float new_vel_y = (d_bodies[id]->vel_y*d_bodies[id]->mass
//                          + d_bodies[m]->vel_y*d_bodies[m]->mass) / new_mass;
//       d_bodies[m]->mass = new_mass;
//       d_bodies[m]->vel_x = new_vel_x;
//       d_bodies[m]->vel_y = new_vel_y;
//       d_bodies[m]->pos_x = (d_bodies[id]->pos_x + d_bodies[m]->pos_x) / 2;
//       d_bodies[m]->pos_y = (d_bodies[id]->pos_y + d_bodies[m]->pos_y) / 2;

//       d_bodies[id]->successful_merge = true;
//     }
//   }
// }

__device__ void Body_update_merge(IndexT id) {
    void **vtable;
    BodyType *ptrId = d_bodies[id];
    COAL_BodyType_get_merge_target(ptrId);
    IndexT m = CLEANPTR(ptrId,BodyType *)->get_merge_target();

    BodyType *ptrm = d_bodies[m];
    if (m != kNullptr) {
        COAL_BodyType_get_merge_target(ptrm);
        if (CLEANPTR(ptrm,BodyType *)->get_merge_target() == kNullptr) {
            // Perform merge.
            COAL_BodyType_get_mass(ptrId);
            float idmass = CLEANPTR(ptrId,BodyType *)->get_mass();
            COAL_BodyType_get_mass(ptrm);
            float mmass = CLEANPTR(ptrm,BodyType *)->get_mass();

            float new_mass = idmass + mmass;
            COAL_BodyType_VelX(ptrId);
            float idvelx = CLEANPTR(ptrId,BodyType *)->VelX();
            COAL_BodyType_VelY(ptrId);
            float idvely = CLEANPTR(ptrId,BodyType *)->VelY();

            COAL_BodyType_VelX(ptrm);
            float mvelx = CLEANPTR(ptrm,BodyType *)->VelX();
            COAL_BodyType_VelY(ptrm);
            float mvely = CLEANPTR(ptrm,BodyType *)->VelY();

            float new_vel_x = (idvelx * idmass + mvelx * mmass) / new_mass;
            float new_vel_y = (idvely * idmass + mvely * mmass) / new_mass;

            COAL_BodyType_set_mass(ptrm);
            CLEANPTR(ptrm,BodyType *)->set_mass(new_mass);
            COAL_BodyType_set_VelX(ptrm);
            CLEANPTR(ptrm,BodyType *)->set_VelX(new_vel_x);
            COAL_BodyType_set_VelY(ptrm);
            CLEANPTR(ptrm,BodyType *)->set_VelY(new_vel_y);
            COAL_BodyType_PosX(ptrId);
            float idposx = CLEANPTR(ptrId,BodyType *)->PosX();
            COAL_BodyType_PosY(ptrId);
            float idposy = CLEANPTR(ptrId,BodyType *)->PosY();
            COAL_BodyType_PosX(ptrm);
            float mposx = CLEANPTR(ptrm,BodyType *)->PosX();
            COAL_BodyType_PosY(ptrm);
            float mposy = CLEANPTR(ptrm,BodyType *)->PosY();

            COAL_BodyType_set_PosX(ptrm);
            CLEANPTR(ptrm,BodyType *)->set_PosX((idposx + mposx) / 2);
            COAL_BodyType_set_PosY(ptrm);
            CLEANPTR(ptrm,BodyType *)->set_PosY((idposy + mposy) / 2);
            COAL_BodyType_set_is_successful_merge(ptrId);
            CLEANPTR(ptrId,BodyType *)->set_is_successful_merge(true);
        }
    }
}

__device__ void Body_delete_merged(IndexT id) {
    void **vtable;
    COAL_BodyType_get_is_successful_merge(d_bodies[id]);
    if (CLEANPTR(d_bodies[id],BodyType *)->get_is_successful_merge()) {
      COAL_BodyType_set_active(d_bodies[id]);
      CLEANPTR(d_bodies[id],BodyType *)->set_active(false);
    }
}

__device__ void Body_add_to_draw_array(IndexT id) {
    int idx = atomicAdd(&r_draw_counter, 1);
    r_Body_pos_x[idx] = CLEANPTR(d_bodies[id],BodyType *)->pos_x;
    r_Body_pos_y[idx] = CLEANPTR(d_bodies[id],BodyType *)->pos_y;
    r_Body_vel_x[idx] = CLEANPTR(d_bodies[id],BodyType *)->vel_x;
    r_Body_vel_y[idx] = CLEANPTR(d_bodies[id],BodyType *)->vel_y;
    r_Body_mass[idx] = CLEANPTR(d_bodies[id],BodyType *)->mass;
}

__device__ void new_Body(IndexT id, float pos_x, float pos_y, float vel_x,
                         float vel_y, float mass) {
    CLEANPTR(d_bodies[id],BodyType *)->pos_x = pos_x;
    CLEANPTR(d_bodies[id],BodyType *)->pos_y = pos_y;
    CLEANPTR(d_bodies[id],BodyType *)->vel_x = vel_x;
    CLEANPTR(d_bodies[id],BodyType *)->vel_y = vel_y;
    CLEANPTR(d_bodies[id],BodyType *)->mass = mass;
    CLEANPTR(d_bodies[id],BodyType *)->is_active = true;
}
__global__ void kernel_initialize_bodies() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    curandState rand_state;
    curand_init(kSeed, tid, 0, &rand_state);
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
        // d_bodies[id] = new Body();
        // assert(d_bodies[id] != NULL);

        new_Body(id,
                 /*pos_x=*/2 * curand_uniform(&rand_state) - 1,
                 /*pos_y=*/2 * curand_uniform(&rand_state) - 1,
                 /*vel_x=*/(curand_uniform(&rand_state) - 0.5) / 1000,
                 /*vel_y=*/(curand_uniform(&rand_state) - 0.5) / 1000,
                 /*mass=*/(curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass);
    }
}

__global__ void kernel_reset_draw_counters() { r_draw_counter = 0; }

template <void (*func)(IndexT)>
__global__ void parallel_do() {
    void **vtable;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
      COAL_BodyType_active(d_bodies[id]);
        if (CLEANPTR(d_bodies[id],BodyType *)->active()) {
            func(id);
        }
    }
}

template <void (*func)(IndexT)>
__global__ void parallel_init() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
        if (d_bodies[id]->active()) {
            func(id);
        }
    }
}

void transfer_data() {
    // Extract data from SoaAlloc data structure.
    kernel_reset_draw_counters<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());
    parallel_init<&Body_add_to_draw_array><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    // Copy data to host.
    cudaMemcpyFromSymbol(host_Body_pos_x, r_Body_pos_x,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_pos_y, r_Body_pos_y,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_vel_x, r_Body_vel_x,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_vel_y, r_Body_vel_y,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_mass, r_Body_mass,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_draw_counter, r_draw_counter, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
}

int checksum() {
    transfer_data();
    int result = 0;

    for (int i = 0; i < kNumBodies; ++i) {
        int Body_checksum =
            static_cast<int>(
                (host_Body_pos_x[i] * 1000 + host_Body_pos_y[i] * 2000 +
                 host_Body_vel_x[i] * 3000 + host_Body_vel_y[i] * 4000)) %
            123456;
        result += Body_checksum;
    }

    return result;
}

int main(int argc, char ** argv) {
#ifdef OPTION_RENDER
    init_renderer();
#endif  // OPTION_RENDER

    // Allocate memory.
    mem_alloc shared_mem(4ULL * 1024 * 1024 * 1024);
    obj_alloc my_obj_alloc(&shared_mem, atoll(argv[1]));
    d_bodies = (BodyType **)my_obj_alloc.calloc<BodyType *>(kNumBodies);

    for (int i = 0; i < kNumBodies; i++) {
        d_bodies[i] = (Body *)my_obj_alloc.my_new<Body>();
    }
    my_obj_alloc.toDevice();
    my_obj_alloc.create_table();
    vfun_table = my_obj_alloc.get_vfun_table();
    // Allocate and create Body objects.
    kernel_initialize_bodies<<<128, 128>>>();
    gpuErrchk(cudaDeviceSynchronize());


    auto time_start = std::chrono::system_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        printf("%i\n", i);
        parallel_do<&Body_compute_force><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
        parallel_do<&Body_update><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
        parallel_do<&Body_initialize_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
        parallel_do<&Body_prepare_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
        parallel_do<&Body_update_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
        parallel_do<&Body_delete_merged><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());
    }

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

#ifndef NDEBUG
    printf("Checksum: %i\n", checksum());
    printf("#bodies: %i\n", host_draw_counter);
#endif  // NDEBUG

    printf("%lu\n", micros);

    // Free memory

#ifdef OPTION_RENDER
    close_renderer();
#endif  // OPTION_RENDER

    return 0;
}
