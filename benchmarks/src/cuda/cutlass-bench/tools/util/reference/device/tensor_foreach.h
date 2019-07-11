/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <stdexcept>
#include "cutlass/cutlass.h"
#include "tools/util/reference/device/kernel/tensor_foreach.h"

namespace cutlass  {
namespace reference {
namespace device {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Launches a kernel for each element in a tensor's index space.
template <typename Func, int Rank, typename Params>
struct TensorForEach {

  /// Constructor performs the operation.
  TensorForEach(Coord<Rank> size, Params params = Params(), int grid_size = 0, int block_size = 0) {

    if (!grid_size || !block_size) {

      // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
      cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
        &grid_size,
        &block_size,
        reinterpret_cast<void const *>(kernel::TensorForEach<Func, Rank, Params>));

      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to query occupancy.");
      }

      // Limit block size. This has the effect of increasing the number of items processed by a
      // single thread and reduces the impact of initialization overhead.
      block_size = (block_size < 128 ? block_size : 128);
    }

    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);

    kernel::TensorForEach<Func, Rank, Params><<< grid, block >>>(size, params);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namesace cutlass
