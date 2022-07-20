/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <CL/sycl.hpp>
#include "common.h"

// include kernels
#include "downscaleKernel.hpp"
#include "upscaleKernel.hpp"
#include "warpingKernel.hpp"
#include "derivativesKernel.hpp"
#include "solverKernel.hpp"
#include "addKernel.hpp"

#include <chrono>

using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;
using float_ms = std::chrono::duration<float, ms::period>;

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  I0           source image
/// \param[in]  I1           tracked image
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  nLevels      number of levels in a pyramid
/// \param[in]  nWarpIters   number of warping iterations per pyramid level
/// \param[in]  nSolverIters number of solver iterations (Jacobi iterations)
/// \param[out] u            horizontal displacement
/// \param[out] v            vertical displacement
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(const float *I0, const float *I1, int width, int height,
                     int stride, float alpha, int nLevels, int nWarpIters,
                     int nSolverIters, float *u, float *v) {
  
  auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during ASUM:\n"
                          << e.what() << std::endl;
                //print_error_code(e);
            }
        }
    };

  queue q{default_selector(), exception_handler, property::queue::in_order()};
  printf("Computing optical flow on GPU...\n");
   std::cout << "\nRunning on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // pI0 and pI1 will hold device pointers
  const float **pI0 = new const float *[nLevels];
  const float **pI1 = new const float *[nLevels];

  int *pW = new int[nLevels];
  int *pH = new int[nLevels];
  int *pS = new int[nLevels];

  // device memory pointers
  float *d_tmp;
  float *d_du0;
  float *d_dv0;
  float *d_du1;
  float *d_dv1;

  float *d_Ix;
  float *d_Iy;
  float *d_Iz;

  float *d_u;
  float *d_v;
  float *d_nu;
  float *d_nv;

  const int dataSize = stride * height * sizeof(float);

  d_tmp = (float *)sycl::malloc_device(dataSize, q);
  d_du0 = (float *)sycl::malloc_device(dataSize, q);
  d_dv0 = (float *)sycl::malloc_device(dataSize, q);
  d_du1 = (float *)sycl::malloc_device(dataSize, q);
  d_dv1 = (float *)sycl::malloc_device(dataSize, q);
  d_Ix  = (float *)sycl::malloc_device(dataSize, q);
  d_Iy  = (float *)sycl::malloc_device(dataSize, q);
  d_Iz  = (float *)sycl::malloc_device(dataSize, q);
  d_u   = (float *)sycl::malloc_device(dataSize, q);
  d_v   = (float *)sycl::malloc_device(dataSize, q);
  d_nu  = (float *)sycl::malloc_device(dataSize, q);
  d_nv  = (float *)sycl::malloc_device(dataSize, q);

  // prepare pyramid

  int currentLevel = nLevels - 1;
  // allocate GPU memory for input images
  *(pI0 + currentLevel) = (const float *)sycl::malloc_device(dataSize, q);
  *(pI1 + currentLevel) = (const float *)sycl::malloc_device(dataSize, q);

  q.memcpy((void *)pI0[currentLevel], I0, dataSize).wait();
  q.memcpy((void *)pI1[currentLevel], I1, dataSize).wait();

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;

  auto down_duration = 0.0;
  auto warp_duration = 0.0;
  auto deriv_duration = 0.0;
  auto solve_duration = 0.0;
  auto up_duration = 0.0;
  auto add_duration = 0.0;
  
  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);

    *(pI0 + currentLevel - 1) = (const float *)sycl::malloc_device(ns * nh * sizeof(float), q);
    *(pI1 + currentLevel - 1) = (const float *)sycl::malloc_device(ns * nh * sizeof(float), q);
    
    auto startDownTime = Time::now();

    Downscale(pI0[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1], q);

    Downscale(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1], q);
    q.wait();
    
    auto stopDownTime = Time::now();
    
    down_duration += std::chrono::duration_cast<float_ms>(stopDownTime - startDownTime)
          .count();

    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }
 // printf("Processing time of Downscale: %f (ms)\n", down_duration);
  
  q.memset(d_u, 0, stride * height * sizeof(float)).wait();
  q.memset(d_v, 0, stride * height * sizeof(float)).wait();

  // compute flow 
  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      q.memset(d_du0, 0, dataSize).wait();
      q.memset(d_dv0, 0, dataSize).wait();

      q.memset(d_du1, 0, dataSize).wait();
      q.memset(d_dv1, 0, dataSize).wait();

      // on current level we compute optical flow
      // between frame 0 and warped frame 1
      auto startWarpTime = Time::now();
      WarpImage(pI1[currentLevel], pW[currentLevel], pH[currentLevel],
                pS[currentLevel], d_u, d_v, d_tmp, q);
      
      auto stopWarpTime = Time::now();
      warp_duration += std::chrono::duration_cast<float_ms>(stopWarpTime - startWarpTime)
          .count();

      auto startDerivTime = Time::now();
      ComputeDerivatives(pI0[currentLevel], d_tmp, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz, q);
      auto stopDerivTime = Time::now();
      deriv_duration += std::chrono::duration_cast<float_ms>(stopDerivTime - startDerivTime)
          .count();
      
      for (int iter = 0; iter < nSolverIters; ++iter) {
        auto startSolveTime = Time::now();
        SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
                       pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1, q);

        auto stopSolveTime = Time::now();
         solve_duration += std::chrono::duration_cast<float_ms>(stopSolveTime - startSolveTime)
          .count();
        Swap(d_du0, d_du1);
        Swap(d_dv0, d_dv1);
      }

      // update u, v
      auto startAddTime = Time::now();
      Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u, q);
      Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v, q);
      
      auto stopAddTime = Time::now();
       add_duration += std::chrono::duration_cast<float_ms>(stopAddTime - startAddTime)
          .count();
    }

    if (currentLevel != nLevels - 1) {
      // prolongate solution
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];
      
      auto startUpTime = Time::now();
      Upscale(d_u, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, d_nu, q);
      
      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(d_v, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, d_nv, q);
      
      auto stopUpTime = Time::now();
       up_duration += std::chrono::duration_cast<float_ms>(stopUpTime - startUpTime)
          .count();

      Swap(d_u, d_nu);
      Swap(d_v, d_nv);
    }
  }

/*  printf("Processing time of WarpImage: %f (ms)\n", warp_duration);
  printf("Processing time of ComputeDerivative: %f (ms)\n", deriv_duration);
  printf("Processing time of SolveforUpdate: %f (ms)\n", solve_duration);
  printf("Processing time of Upscale: %f (ms)\n", up_duration);
  printf("Processing time of Add: %f (ms)\n", add_duration);
  */
  
  q.memcpy(u, d_u, dataSize).wait();
  q.memcpy(v, d_v, dataSize).wait();

  // cleanup
  for (int i = 0; i < nLevels; ++i) {
    sycl::free((void *)pI0[i], q);
    sycl::free((void *)pI1[i], q);
  }

  delete[] pI0;
  delete[] pI1;
  delete[] pW;
  delete[] pH;
  delete[] pS;

  sycl::free(d_tmp, q);
  sycl::free(d_du0, q);
  sycl::free(d_dv0, q);
  sycl::free(d_du1, q);
  sycl::free(d_dv1, q);
  sycl::free(d_Ix, q);
  sycl::free(d_Iy, q);
  sycl::free(d_Iz, q);
  sycl::free(d_nu, q);
  sycl::free(d_nv, q);
  sycl::free(d_u, q);
  sycl::free(d_v, q);
}
