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
/// \brief upscale one component of a displacement field, CUDA kernel
/// \param[in]  width   field width
/// \param[in]  height  field height
/// \param[in]  stride  field stride
/// \param[in]  scale   scale factor (multiplier)
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
void UpscaleKernel(int width, int height, int stride, float scale, float *out,
dpct::image_accessor_ext<cl::sycl::float4, 2> texCoarse,
                   sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);

  if (ix >= width || iy >= height) return;

  float x = ((float)ix + 0.5f) / (float)width;
  float y = ((float)iy + 0.5f) / (float)height;

  // exploit hardware interpolation
  // and scale interpolated vector to match next pyramid level resolution
  out[ix + iy * stride] = texCoarse.read(x, y)[0] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  stride      field current stride
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  newStride   field new stride
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride, float scale,
                    float *out, queue q) {
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));
  
  int dataSize = stride * height * sizeof(float);
  float *src_h = (float *)malloc(dataSize);
  q.memcpy(src_h, src, dataSize).wait();
  
  float *src_p = (float *)sycl::malloc_shared(4 * dataSize, q);
  for (int i=0; i < height; i++) {
    for (int j = 0; j< width; j++){
      int index = i * stride + j;
    src_p[index * 4 + 0] = src_h[index];
    src_p[index * 4 + 1] = src_p[index * 4 + 2] = src_p[index * 4 + 3] = 0.f;
  }
  }
  dpct::image_wrapper_base_p texCoarse;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)src_p);
  /*
  DPCT1059:5: SYCL only supports 4-channel image format. Adjust the code.
  */
  texRes.set_channel(dpct::image_channel::create<sycl::float4>());
  texRes.set_x(width);
  texRes.set_y(height);
  texRes.set_pitch(stride * sizeof(float));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::mirrored_repeat,
               sycl::filtering_mode::linear,
               sycl::coordinate_normalization_mode::normalized);

  texCoarse = dpct::create_image_wrapper(texRes, texDescr);

  /*
  DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  q.submit([&](sycl::handler &cgh) {
    auto texCoarse_acc =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texCoarse)->get_access(
            cgh);

    auto texCoarse_smpl = texCoarse->get_sampler();

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpscaleKernel(newWidth, newHeight, newStride, scale, out,
                                     dpct::image_accessor_ext<cl::sycl::float4, 2>(
                                         texCoarse_smpl, texCoarse_acc),
                                     item_ct1);
                     });
  }).wait();
}
