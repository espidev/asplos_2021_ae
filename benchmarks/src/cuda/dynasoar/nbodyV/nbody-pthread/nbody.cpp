#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <fenv.h>
 
using namespace std;
// #include <chrono>

#define ALL __attribute__((noinline))

#include "../configuration.h"

#define NUM_THREADS 16

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
    __attribute__((noinline)) virtual void initBody(int idx) = 0;
    ALL BodyType() {}
    __attribute__((noinline)) BodyType(int idx) {}

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
    __attribute__((noinline)) void initBody(int idx) {
        // curandState rand_state;
        // curand_init(kSeed, idx, 0, &rand_state);

        // pos_x = 2 * curand_uniform(&rand_state) - 1;
        // pos_y = 2 * curand_uniform(&rand_state) - 1;
        // vel_x = (curand_uniform(&rand_state) - 0.5) / 1000;
        // vel_y = (curand_uniform(&rand_state) - 0.5) / 1000;
        // mass = (curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass;
    }
    Body(int idx) {
        // curandState rand_state;
        // curand_init(kSeed, idx, 0, &rand_state);

        // pos_x = 2 * curand_uniform(&rand_state) - 1;
        // pos_y = 2 * curand_uniform(&rand_state) - 1;
        // vel_x = (curand_uniform(&rand_state) - 0.5) / 1000;
        // vel_y = (curand_uniform(&rand_state) - 0.5) / 1000;
        // mass = (curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass;
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

float device_checksum;

struct args {
    int tid;
    BodyType **dev_bodies;
};

void *Body_compute_force(void *ar) {
    struct args *arPtr = (struct args *)ar;

    int tid = arPtr->tid;
    BodyType **dev_bodies = arPtr->dev_bodies;
    float dist;
    float F;
    for (int id = tid; id < kNumBodies; id += NUM_THREADS) {
        dev_bodies[id]->initForce();
        // printf("%d ddd\n", id);
        for (int i = 0; i < kNumBodies; ++i) {
            // Do not compute force with the body itself.
            if (id != i) {
                dist = dev_bodies[id]->computeDistance(dev_bodies[i]);
                F = dev_bodies[id]->computeForce(dev_bodies[i], dist);
                dev_bodies[id]->updateForceX(dev_bodies[i], F);
                dev_bodies[id]->updateForceY(dev_bodies[i], F);
            }
        }
    }
    return NULL;
}

void *Body_update(void *ar) {
    struct args *arPtr = (struct args *)ar;

    int tid = arPtr->tid;
    BodyType **dev_bodies = arPtr->dev_bodies;

    for (int id = tid; id < kNumBodies; id += NUM_THREADS) {
        dev_bodies[id]->updateVelX();
        dev_bodies[id]->updateVelY();
        dev_bodies[id]->updatePosX();
        dev_bodies[id]->updatePosY();

        if (dev_bodies[id]->pos_x < -1 || dev_bodies[id]->pos_x > 1) {
            dev_bodies[id]->vel_x = -dev_bodies[id]->vel_x;
        }

        if (dev_bodies[id]->pos_y < -1 || dev_bodies[id]->pos_y > 1) {
            dev_bodies[id]->vel_y = -dev_bodies[id]->vel_y;
        }
    }
    return NULL;
}

void Body_add_checksum(BodyType **dev_bodies, int id) {
    // atomicAdd(&device_checksum,
    //           dev_bodies[id]->pos_x + dev_bodies[id]->pos_y * 2 +
    //               dev_bodies[id]->vel_x * 3 + dev_bodies[id]->vel_y * 4);
}

void kernel_initialize_bodies(BodyType **bodies) {
    for (int i = 0; i < kNumBodies; i++) {
        bodies[i] = new Body(/*idx*/ i);
        bodies[i]->initBody(i);
    }
}

void kernel_compute_checksum(BodyType **bodies) {
    device_checksum = 0.0f;
    for (int i = 0; i < kNumBodies; ++i) {
        Body_add_checksum(bodies, i);
    }
}

// void print_ptr_diff(BodyType **ptr) {
//     int i;
//     for (i = 1; i < kNumBodies / 100; i++) {
//       unsigned long long ptr2=(unsigned long long)ptr[i];
//       unsigned long long ptr1=(unsigned long long)ptr[i-1];
//         printf("[ptr[%d]-ptr[%d]]= %ull\n", i, i - 1,
//     (ptr2-  ptr1 ));
//     }
//   }

int main(int /*argc*/, char ** /*argv*/) {
    BodyType **dev_bodies;

    // Allocate and create Body objects.
    dev_bodies = (BodyType **)malloc(sizeof(BodyType *) * kNumBodies);
    printf("init bodies...\n");
    kernel_initialize_bodies(dev_bodies);

    pthread_t threads[NUM_THREADS];
    struct args arg[NUM_THREADS];
    pthread_attr_t attr;
    void *status;
    int i;
    int rc;
    // Initialize and set thread joinable
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    //     gpuErrchk(cudaDeviceSynchronize());

    //     printf("init done...\n");
    //     auto time_start = std::chrono::system_clock::now();
    //     printf("Kernel exec...\n");
    for (int j = 0; j < kNumIterations; ++j) {
         printf("Start: BodyComputeForce(%d)\n", j);

        for (i = 0; i < NUM_THREADS; i++) {
            arg[i].tid = i;
            arg[i].dev_bodies = dev_bodies;
            //   cout << "main() : creating thread, " << i << endl;
            rc = pthread_create(&threads[i], &attr, Body_compute_force,
                                (void *)&arg[i]);
            if (rc) {
                cout << "Error:unable to create thread," << rc << endl;
                exit(-1);
            }
        }
        // printf("Body_compute_force(%d)\n",i);
        for (i = 0; i < NUM_THREADS; i++) {
            rc = pthread_join(threads[i], &status);
            if (rc) {
                cout << "Error:unable to join," << rc << endl;
                exit(-1);
            }
            // cout << "Main: completed thread id :" << i;
            // cout << "  exiting with status :" << status << endl;
        }

        for (i = 0; i < NUM_THREADS; i++) {
            arg[i].tid = i;
            arg[i].dev_bodies = dev_bodies;
            //   cout << "main() : creating thread, " << i << endl;
            rc = pthread_create(&threads[i], &attr, Body_update,
                                (void *)&arg[i]);
            if (rc) {
                cout << "Error:unable to create thread," << rc << endl;
                exit(-1);
            }
        }
        for (i = 0; i < NUM_THREADS; i++) {
            rc = pthread_join(threads[i], &status);
            if (rc) {
                cout << "Error:unable to join," << rc << endl;
                exit(-1);
            }
            // cout << "Main: completed thread id :" << i;
            // cout << "  exiting with status :" << status << endl;
        }

        if (j % 300 == 0) printf("Finish: BodyComputeForce(%d)\n", j);
    }

    //     pthread_attr_destroy(&attr);

    //     auto time_end = std::chrono::system_clock::now();
    //     auto elapsed = time_end - time_start;
    //     auto micros =
    //         std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    //     printf("%lu\n", micros);

    // #ifndef NDEBUG
    //     kernel_compute_checksum<<<1, 1>>>(dev_bodies);
    //     gpuErrchk(cudaDeviceSynchronize());

    //     float checksum;
    //     cudaMemcpyFromSymbol(&checksum, device_checksum,
    //     sizeof(device_checksum), 0,
    //                          cudaMemcpyDeviceToHost);
    //     printf("Checksum: %f\n", checksum);
    // #endif  // NDEBUG

    //     cudaFree(dev_bodies);

    return 0;
}
