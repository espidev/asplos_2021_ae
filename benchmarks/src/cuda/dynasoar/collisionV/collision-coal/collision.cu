
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
        float dist = ptr->computeDistance(d_bodies[id]);
        COAL_BodyType_computeForce(ptr);
        float F = ptr->computeForce(d_bodies[id], dist);
        COAL_BodyType_updateForceX(ptr);
        ptr->updateForceX(d_bodies[id], F);
        COAL_BodyType_updateForceY(ptr);
        ptr->updateForceY(d_bodies[id], F);
    }
}

__device__ void Body_compute_force(IndexT id) {
    void **vtable;
    COAL_BodyType_initForce(d_bodies[id]);
    d_bodies[id]->initForce();

    // device_do
    for (IndexT i = 0; i < kNumBodies; ++i) {
        COAL_BodyType_active(d_bodies[i]);
        if (d_bodies[i]->active()) {
            Body_apply_force(i, id);
        }
    }
}

__device__ void Body_update(IndexT id) {
    void **vtable;
    BodyType *ptr = d_bodies[id];
    COAL_BodyType_updateVelX(ptr);
    ptr->updateVelX();
    COAL_BodyType_updateVelY(ptr);
    ptr->updateVelY();
    COAL_BodyType_updatePosX(ptr);
    ptr->updatePosX();
    COAL_BodyType_updatePosX(ptr);
    ptr->updatePosY();
    COAL_BodyType_PosX(ptr);
    float idposx = ptr->PosX();
    COAL_BodyType_PosY(ptr);
    float idposy = ptr->PosY();
    COAL_BodyType_VelX(ptr);
    float idvelx = ptr->VelX();
    COAL_BodyType_VelY(ptr);
    float idvely = ptr->VelY();

    if (idposx < -1 || idposx > 1) {
        COAL_BodyType_set_VelX(ptr);
        ptr->set_VelX(-idvelx);
    }

    if (idposy < -1 || idposy > 1) {
        COAL_BodyType_set_VelY(ptr);
        ptr->set_VelY(-idvely);
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
    bool cond1 = otherPtr->get_incoming_merge();
    COAL_BodyType_get_mass(otherPtr);
    float othermass = otherPtr->get_mass();
    COAL_BodyType_get_mass(otherPtr);
    float idmass = d_bodies[id]->get_mass();

    if (!cond1 && idmass > othermass) {
        COAL_BodyType_computeDistance(d_bodies[id]);
        float dist_square = d_bodies[id]->computeDistance(otherPtr);
        dist_square *= dist_square;
        if (dist_square < kMergeThreshold * kMergeThreshold) {
            // Try to merge this one.
            // There is a race condition here: Multiple threads may try to merge
            // this body.
            COAL_BodyType_set_merge_target(d_bodies[id]);
            d_bodies[id]->set_merge_target(other);
            COAL_BodyType_set_incoming_merge(otherPtr);
            otherPtr->set_incoming_merge(true);
        }
    }
}

__device__ void Body_initialize_merge(IndexT id) {
    void **vtable;
    BodyType *ptr = d_bodies[id];
    COAL_BodyType_set_merge_target(ptr);
    ptr->set_merge_target(kNullptr);
    COAL_BodyType_set_incoming_merge(ptr);
    ptr->set_incoming_merge(false);
    COAL_BodyType_set_is_successful_merge(ptr);
    ptr->set_is_successful_merge(false);
}

__device__ void Body_prepare_merge(IndexT id) {
    // device_do
    void **vtable;
    for (IndexT i = 0; i < kNumBodies; ++i) {
        COAL_BodyType_active(d_bodies[i]);
        if (d_bodies[i]->active()) {
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
    IndexT m = ptrId->get_merge_target();

    BodyType *ptrm = d_bodies[m];
    if (m != kNullptr) {
        COAL_BodyType_get_merge_target(ptrm);
        if (ptrm->get_merge_target() == kNullptr) {
            // Perform merge.
            COAL_BodyType_get_mass(ptrId);
            float idmass = ptrId->get_mass();
            COAL_BodyType_get_mass(ptrm);
            float mmass = ptrm->get_mass();

            float new_mass = idmass + mmass;
            COAL_BodyType_VelX(ptrId);
            float idvelx = ptrId->VelX();
            COAL_BodyType_VelY(ptrId);
            float idvely = ptrId->VelY();

            COAL_BodyType_VelX(ptrm);
            float mvelx = ptrm->VelX();
            COAL_BodyType_VelY(ptrm);
            float mvely = ptrm->VelY();

            float new_vel_x = (idvelx * idmass + mvelx * mmass) / new_mass;
            float new_vel_y = (idvely * idmass + mvely * mmass) / new_mass;

            COAL_BodyType_set_mass(ptrm);
            ptrm->set_mass(new_mass);
            COAL_BodyType_set_VelX(ptrm);
            ptrm->set_VelX(new_vel_x);
            COAL_BodyType_set_VelY(ptrm);
            ptrm->set_VelY(new_vel_y);
            COAL_BodyType_PosX(ptrId);
            float idposx = ptrId->PosX();
            COAL_BodyType_PosY(ptrId);
            float idposy = ptrId->PosY();
            COAL_BodyType_PosX(ptrm);
            float mposx = ptrm->PosX();
            COAL_BodyType_PosY(ptrm);
            float mposy = ptrm->PosY();

            COAL_BodyType_set_PosX(ptrm);
            ptrm->set_PosX((idposx + mposx) / 2);
            COAL_BodyType_set_PosY(ptrm);
            ptrm->set_PosY((idposy + mposy) / 2);
            COAL_BodyType_set_is_successful_merge(ptrId);
            ptrId->set_is_successful_merge(true);
        }
    }
}

__device__ void Body_delete_merged(IndexT id) {
    void **vtable;
    COAL_BodyType_get_is_successful_merge(d_bodies[id]);
    if (d_bodies[id]->get_is_successful_merge()) {
      COAL_BodyType_set_active(d_bodies[id]);
        d_bodies[id]->set_active(false);
    }
}

__device__ void Body_add_to_draw_array(IndexT id) {
    int idx = atomicAdd(&r_draw_counter, 1);
    r_Body_pos_x[idx] = d_bodies[id]->pos_x;
    r_Body_pos_y[idx] = d_bodies[id]->pos_y;
    r_Body_vel_x[idx] = d_bodies[id]->vel_x;
    r_Body_vel_y[idx] = d_bodies[id]->vel_y;
    r_Body_mass[idx] = d_bodies[id]->mass;
}

__device__ void new_Body(IndexT id, float pos_x, float pos_y, float vel_x,
                         float vel_y, float mass) {
    d_bodies[id]->pos_x = pos_x;
    d_bodies[id]->pos_y = pos_y;
    d_bodies[id]->vel_x = vel_x;
    d_bodies[id]->vel_y = vel_y;
    d_bodies[id]->mass = mass;
    d_bodies[id]->is_active = true;
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
        if (d_bodies[id]->active()) {
            func(id);
        }
    }
}

void transfer_data() {
    // Extract data from SoaAlloc data structure.
    kernel_reset_draw_counters<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());
    parallel_do<&Body_add_to_draw_array><<<kBlocks, kThreads>>>();
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
    my_obj_alloc.create_tree();
    range_tree = my_obj_alloc.get_range_tree();
    tree_size = my_obj_alloc.get_tree_size();
    // Allocate and create Body objects.
    kernel_initialize_bodies<<<128, 128>>>();
    gpuErrchk(cudaDeviceSynchronize());

#ifdef OPTION_RENDER
    // Compute max_mass.
    float max_mass = 0.0f;
    transfer_data();

    for (int i = 0; i < host_draw_counter; ++i) {
        max_mass += host_Body_mass[i];
    }
#endif  // OPTION_RENDER

    auto time_start = std::chrono::system_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        printf("%i\n", i);
        parallel_do<&Body_compute_force><<<kBlocks, kThreads>>>();

        parallel_do<&Body_update><<<kBlocks, kThreads>>>();

        parallel_do<&Body_initialize_merge><<<kBlocks, kThreads>>>();

        parallel_do<&Body_prepare_merge><<<kBlocks, kThreads>>>();

        parallel_do<&Body_update_merge><<<kBlocks, kThreads>>>();

        parallel_do<&Body_delete_merged><<<kBlocks, kThreads>>>();
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
