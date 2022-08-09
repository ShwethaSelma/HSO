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


using namespace sycl;


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
void ComputeFlowSYCL(const float *I0, const float *I1, int width, int height,
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
  *(pI0 + currentLevel) = (const float *)sycl::malloc_device(dataSize * 4, q);
  *(pI1 + currentLevel) = (const float *)sycl::malloc_device(dataSize * 4, q);
  
  
  float *pI0_h  = (float *)sycl::malloc_host(stride * height * sizeof(sycl::float4), q);
  float *I0_h   = (float *)sycl::malloc_host(dataSize, q);
  
  float *pI1_h  = (float *)sycl::malloc_host(stride * height * sizeof(sycl::float4), q);
  float *I1_h   = (float *)sycl::malloc_host(dataSize, q);
  
  float *src_d0  = (float *)sycl::malloc_device(stride * height * sizeof(sycl::float4), q);
  float *src_d1  = (float *)sycl::malloc_device(stride * height * sizeof(sycl::float4), q);
  
  q.memcpy((void *)I0_h, I0, dataSize);
  q.memcpy((void *)I1_h, I1, dataSize);

  q.memcpy((void *)pI0[currentLevel], I0, dataSize);
  q.memcpy((void *)pI1[currentLevel], I1, dataSize);
  
  q.wait();

  pW[currentLevel] = width;
  pH[currentLevel] = height;
  pS[currentLevel] = stride;
  
  for (; currentLevel > 0; --currentLevel) {
    int nw = pW[currentLevel] / 2;
    int nh = pH[currentLevel] / 2;
    int ns = iAlignUp(nw);

    *(pI0 + currentLevel - 1) = (const float *)sycl::malloc_device(ns * nh * sizeof(sycl::float4), q);
    *(pI1 + currentLevel - 1) = (const float *)sycl::malloc_device(ns * nh * sizeof(sycl::float4), q);
    
    Downscale(pI0[currentLevel], pI0_h, I0_h, src_d0, pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI0[currentLevel - 1], q);

    Downscale(pI1[currentLevel], pI0_h, I0_h, src_d0, pW[currentLevel], pH[currentLevel],
              pS[currentLevel], nw, nh, ns, (float *)pI1[currentLevel - 1], q);
    
    pW[currentLevel - 1] = nw;
    pH[currentLevel - 1] = nh;
    pS[currentLevel - 1] = ns;
  }
  
  q.memset(d_u, 0, stride * height * sizeof(float));
  q.memset(d_v, 0, stride * height * sizeof(float));
  
  q.wait();

  // compute flow 
  for (; currentLevel < nLevels; ++currentLevel) {
    for (int warpIter = 0; warpIter < nWarpIters; ++warpIter) {
      q.memset(d_du0, 0, dataSize);
      q.memset(d_dv0, 0, dataSize);

      q.memset(d_du1, 0, dataSize);
      q.memset(d_dv1, 0, dataSize);
      
      q.wait();

      // on current level we compute optical flow
      // between frame 0 and warped frame 1
      WarpImage(pI1[currentLevel], pI0_h, I0_h, src_d0, pW[currentLevel], pH[currentLevel],
                pS[currentLevel], d_u, d_v, d_tmp, q);
      
      ComputeDerivatives(pI0[currentLevel], d_tmp, pI0_h, pI1_h, I0_h, I1_h, src_d0, src_d1, pW[currentLevel],
                         pH[currentLevel], pS[currentLevel], d_Ix, d_Iy, d_Iz, q);
      
      for (int iter = 0; iter < nSolverIters; ++iter) {
        SolveForUpdate(d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
                       pH[currentLevel], pS[currentLevel], alpha, d_du1, d_dv1, q);

        Swap(d_du0, d_du1);
        Swap(d_dv0, d_dv1);
      }

      // update u, v
      Add(d_u, d_du0, pH[currentLevel] * pS[currentLevel], d_u, q);
      Add(d_v, d_dv0, pH[currentLevel] * pS[currentLevel], d_v, q);
    }

    if (currentLevel != nLevels - 1) {
      // prolongate solution
      float scaleX = (float)pW[currentLevel + 1] / (float)pW[currentLevel];
      
      Upscale(d_u, pI0_h, I0_h, src_d0, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleX, d_nu, q);
      
      float scaleY = (float)pH[currentLevel + 1] / (float)pH[currentLevel];

      Upscale(d_v, pI0_h, I0_h, src_d0, pW[currentLevel], pH[currentLevel], pS[currentLevel],
              pW[currentLevel + 1], pH[currentLevel + 1], pS[currentLevel + 1],
              scaleY, d_nv, q);
  
      Swap(d_u, d_nu);
      Swap(d_v, d_nv);
    }
  }
  
  q.memcpy(u, d_u, dataSize);
  q.memcpy(v, d_v, dataSize);
  
  q.wait();

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
