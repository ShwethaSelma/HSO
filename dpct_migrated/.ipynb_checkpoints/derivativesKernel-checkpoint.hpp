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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

using namespace sycl;

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////

void ComputeDerivativesKernel(int width, int height, int stride, float *Ix,
                              float *Iy, float *Iz,
                              dpct::image_accessor_ext<cl::sycl::float4, 2> texSource,
                              dpct::image_accessor_ext<cl::sycl::float4, 2> texTarget,
                              sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;
  
  float x = ((float)ix);
  float y = ((float)iy);

  float t0, t1;
  // x derivative
  t0 = texSource.read(x, y)[0];
  t0 -= texSource.read(x, y)[0] * 8.0f;
  t0 += texSource.read(x, y)[0] * 8.0f;
  t0 -= texSource.read(x, y)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x, y)[0];
  t1 -= texTarget.read(x, y)[0] * 8.0f;
  t1 += texTarget.read(x, y)[0] * 8.0f;
  t1 -= texTarget.read(x, y)[0];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1);

  // t derivative
  Iz[pos] = texTarget.read(x, y)[0] - texSource.read(x, y)[0];

  // y derivative
  t0 = texSource.read(x, y)[0];
  t0 -= texSource.read(x, y)[0] * 8.0f;
  t0 += texSource.read(x, y)[0] * 8.0f;
  t0 -= texSource.read(x, y)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x, y)[0];
  t1 -= texTarget.read(x, y)[0] * 8.0f;
  t1 += texTarget.read(x, y)[0] * 8.0f;
  t1 -= texTarget.read(x, y)[0];
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1);
}


///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz, queue q) {
  sycl::range<3> threads(1, 6, 32);
  auto max_wg_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
  if( max_wg_size < threads[1] * threads[2])
  {
    threads[0] = 1;
    threads[2] = 32;
    threads[1] = max_wg_size / threads[2];
  }
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));
  
  int dataSize = s * h * sizeof(float);
  
  float *I0_h = (float *)malloc(dataSize);
  float *I1_h = (float *)malloc(dataSize);
  q.memcpy(I0_h, I0, dataSize).wait();
  q.memcpy(I1_h, I1, dataSize).wait();
  
  float *I0_p = (float *)sycl::malloc_shared(4 * dataSize, q);
  for (int i=0; i < h; i++) {
    for (int j = 0; j< w; j++){
      int index = i * s + j;
    I0_p[index * 4 + 0] = I0_h[index];
    I0_p[index * 4 + 1] = I0_p[index * 4 + 2] = I0_p[index * 4 + 3] = 0.f;
  }
  }
  
  float *I1_p = (float *)sycl::malloc_shared(4 * dataSize, q);
   for (int i=0; i < h; i++) {
    for (int j = 0; j< w; j++){
      int index = i * s + j;
    I1_p[index * 4 + 0] = I1_h[index];
    I1_p[index * 4 + 1] = I1_p[index * 4 + 2] = I1_p[index * 4 + 3] = 0.f;
  }
   }

 dpct::image_wrapper_base_p texSource, texTarget;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I0_p);
  

  texRes.set_channel(dpct::image_channel::create<sycl::float4>());
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(sycl::float4));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::clamp,
               sycl::filtering_mode::nearest,
               sycl::coordinate_normalization_mode::unnormalized);
  
  texSource = dpct::create_image_wrapper(texRes, texDescr);
  
  memset(&texRes, 0, sizeof(dpct::image_data));
  
  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I1_p);
  

  texRes.set_channel(dpct::image_channel::create<sycl::float4>());       
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(sycl::float4));
  texTarget = dpct::create_image_wrapper(texRes, texDescr);
  
  q.submit([&](sycl::handler &cgh) {
    auto texSource_acc =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texSource)->get_access(
            cgh);
    auto texTarget_acc =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texTarget)->get_access(
            cgh);

    auto texSource_smpl = texSource->get_sampler();
    auto texTarget_smpl = texTarget->get_sampler();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          ComputeDerivativesKernel(
              w, h, s, Ix, Iy, Iz,
              dpct::image_accessor_ext<cl::sycl::float4, 2>(texSource_smpl, texSource_acc),
              dpct::image_accessor_ext<cl::sycl::float4, 2>(texTarget_smpl, texTarget_acc),
              item_ct1);
        });
  }).wait();
}

