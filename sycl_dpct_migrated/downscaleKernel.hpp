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
#include <iostream>
#include <dpct/dpct.hpp>
#include "common.h"

using namespace sycl;

///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void DownscaleKernel(int width, int height, int stride, float *out,
                    dpct::image_accessor_ext<cl::sycl::float4, 2> texFine,
                     sycl::nd_item<3> item_ct1, stream stream_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  if (ix >= width || iy >= height) {
    return;
  }

  float x = ((float)ix + 0.5f);
  float y = ((float)iy + 0.5f);

  out[ix + iy * stride] =
     0.25f * (texFine.read(x, y)[0] + texFine.read(x, y)[0] +
       texFine.read(x, y)[0] + texFine.read(x, y)[0]);

}


///////////////////////////////////////////////////////////////////////////////
/// \brief downscale image
///
/// \param[in]  src     image to downscale
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out, queue q) {
  sycl::range<3> threads(1, 8, 32);
  auto max_wg_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
  if( max_wg_size < threads[1] * threads[2])
  {
    threads[0] = 1;
    threads[2] = 32;
    threads[1] = max_wg_size / threads[2];
  }
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));
  
  int dataSize = height * stride * sizeof(float);
  float *src_h = (float *)malloc(dataSize);
  q.memcpy(src_h, src, dataSize).wait();
  
  float *src_p = (float *)sycl::malloc_shared(4 * dataSize, q);
  for (int i=0; i < 4 * height * stride; i++)
    src_p[i] = 0.f;
  
  for (int i=0; i < height; i++) {
    for (int j=0; j < width; j++) {
      int index = i * stride + j;
      src_p[index * 4 + 0] = src_h[index];
      src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;
    }
  }
  
  dpct::image_wrapper_base_p texFine;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)src_p);
  
  texRes.set_channel(dpct::image_channel::create<sycl::float4>());
  texRes.set_x(width);
  texRes.set_y(height);
  texRes.set_pitch(stride * sizeof(sycl::float4));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::clamp,
               sycl::filtering_mode::nearest,
               sycl::coordinate_normalization_mode::unnormalized);
  
 texFine = dpct::create_image_wrapper(texRes, texDescr);
  
  q.submit([&](sycl::handler &cgh) {
    auto texFine_acc =
        static_cast<dpct::image_wrapper<sycl::float4, 2> *>(texFine)->get_access(cgh);

    auto texFine_smpl = texFine->get_sampler();
     stream stream_ct1(128 * 1024, 256, cgh);
    
    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       DownscaleKernel(newWidth, newHeight, newStride, out,
                                       dpct::image_accessor_ext<cl::sycl::float4, 2>(
                                           texFine_smpl, texFine_acc),
                                       item_ct1, stream_ct1);
                     });
  }).wait();
}

