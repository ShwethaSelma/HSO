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
                     accessor<cl::sycl::float4, 2, cl::sycl::access::mode::read, sycl::access::target::image> tex_acc, cl::sycl::sampler texDesc,
                     sycl::nd_item<3> item_ct1,  stream stream_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  if (ix >= width || iy >= height) {
    return;
  }

//   float dx = 1.0f / (float)width;
//   float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) ;
  float y = ((float)iy + 0.5f) ;
  
  auto inputCoords1 = float2(x - 0.25f, y);
  auto inputCoords2 = float2(x + 0.25f, y);
  auto inputCoords3 = float2(x, y - 0.25f);
  auto inputCoords4 = float2(x, y + 0.25f);
  
  out[ix + iy * stride] =
    0.25f *
    (tex_acc.read(inputCoords1, texDesc)[0] + tex_acc.read(inputCoords2, texDesc)[0] + 
     tex_acc.read(inputCoords3, texDesc)[0] + tex_acc.read(inputCoords4, texDesc)[0]);
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
  if( max_wg_size < threads[1]*threads[2])
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

  auto texDescr = cl::sycl::sampler(sycl::coordinate_normalization_mode::unnormalized, 
                                    sycl::addressing_mode::mirrored_repeat, 
                                    sycl::filtering_mode::linear);
  
  auto texFine = cl::sycl::image<2>(src_p, 
                                    cl::sycl::image_channel_order::rgba, 
                                    cl::sycl::image_channel_type::fp32, 
                                    range<2>(width, height), range<1>(stride * 4 * sizeof(float)));
  
  q.submit([&](sycl::handler &cgh) {
    auto tex_acc = texFine.template get_access<cl::sycl::float4, cl::sycl::access::mode::read>(cgh);
    stream stream_ct1(128 * 1024, 256, cgh);
    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       DownscaleKernel(newWidth, newHeight, newStride, out,
                                       tex_acc, texDescr,
                                       item_ct1, stream_ct1);
                     });
  }).wait();
}
