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
/// \brief warp image with a given displacement field, CUDA kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void WarpingKernel(int width, int height, int stride, const float *u,
                   const float *v, float *out,
                 dpct::image_accessor_ext<cl::sycl::float4, 2> texToWarp,
                   sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float x = ((float)ix + u[pos] + 0.5f) / (float)width;
  float y = ((float)iy + v[pos] + 0.5f) / (float)height;

  out[pos] = texToWarp.read(x, y)[0];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, CUDA kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, int w, int h, int s, const float *u,
                      const float *v, float *out, queue q) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));
  
  int dataSize = s * h * sizeof(float);
  float *src_h = (float *)malloc(dataSize);
  q.memcpy(src_h, src, dataSize).wait();
  
  float *src_p = (float *)sycl::malloc_shared(4 * dataSize, q);
  for (int i=0; i < h; i++) {
    for (int j = 0; j< w; j++){
      int index = i * s + j;
    src_p[index * 4 + 0] = src_h[index];
    src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;
  }
  }

   dpct::image_wrapper_base_p texToWarp;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)src_p);
  /*
  DPCT1059:9: SYCL only supports 4-channel image format. Adjust the code.
  */
  texRes.set_channel(dpct::image_channel::create<sycl::float4>());
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * 4 * sizeof(float));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::mirrored_repeat,
               sycl::filtering_mode::linear,
               sycl::coordinate_normalization_mode::normalized);
  /*
  DPCT1004:10: Compatible DPC++ code could not be generated.
  */
  //texDescr.readMode = cudaReadModeElementType;

  texToWarp = dpct::create_image_wrapper(texRes, texDescr);
  
  q.submit([&](sycl::handler &cgh) {
    auto texToWarp_acc =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texToWarp)->get_access(
            cgh);

    auto texToWarp_smpl = texToWarp->get_sampler();

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       WarpingKernel(w, h, s, u, v, out,
                                     dpct::image_accessor_ext<cl::sycl::float4, 2>(
                                         texToWarp_smpl, texToWarp_acc),
                                     item_ct1);
                     });
  }).wait();
}
