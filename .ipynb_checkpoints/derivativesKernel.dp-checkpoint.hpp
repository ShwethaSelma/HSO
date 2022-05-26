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

int convert(int ind) {
  return ((ind/4) + (ind % 4));
}

void ComputeDerivativesKernel1(int width, int height, int stride, float *Ix,
                              float *Iy, float *Iz,
                              dpct::image_accessor_ext<cl::sycl::float4, 2> texSource,
                              dpct::image_accessor_ext<cl::sycl::float4, 2> texTarget,
                              sycl::nd_item<3> item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
  
  // dpct::image_accessor_ext<float, 2> *textSource = static_cast<dpct::image_accessor_ext<float, 2> *>(&texSource1);
  // dpct::image_accessor_ext<float, 2> *textTarget = static_cast<dpct::image_accessor_ext<float, 2> *>(&texTarget1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  float t0, t1;
  // x derivative
  t0 = texSource.read(convert(x - 2.0f * dx), convert(y))[0];
  t0 -= texSource.read(convert(x - 1.0f * dx), convert(y))[0] * 8.0f;
  t0 += texSource.read(convert(x + 1.0f * dx), convert(y))[0] * 8.0f;
  t0 -= texSource.read(convert(x + 2.0f * dx), convert(y))[0];
  t0 /= 12.0f;

  t1 = texTarget.read(convert(x - 2.0f * dx), convert(y))[0];
  t1 -= texTarget.read(convert(x - 1.0f * dx), convert(y))[0] * 8.0f;
  t1 += texTarget.read(convert(x + 1.0f * dx), convert(y))[0] * 8.0f;
  t1 -= texTarget.read(convert(x + 2.0f * dx), convert(y))[0];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  Iz[pos] = texTarget.read(convert(x), convert(y))[0] - texSource.read(convert(x), convert(y))[0];

  // y derivative
  t0 = texSource.read(convert(x), convert(y - 2.0f * dy))[0];
  t0 -= texSource.read(convert(x), convert(y - 1.0f * dy))[0] * 8.0f;
  t0 += texSource.read(convert(x), convert(y + 1.0f * dy))[0] * 8.0f;
  t0 -= texSource.read(convert(x), convert(y + 2.0f * dy))[0];
  t0 /= 12.0f;

  t1 = texTarget.read(convert(x), convert(y - 2.0f * dy))[0];
  t1 -= texTarget.read(convert(x), convert(y - 1.0f * dy))[0] * 8.0f;
  t1 += texTarget.read(convert(x), convert(y + 1.0f * dy))[0] * 8.0f;
  t1 -= texTarget.read(convert(x), convert(y + 2.0f * dy))[0];
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;
}

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

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  float t0, t1;
  // x derivative
  t0 = texSource.read(x - 2.0f * dx, y)[0];
  t0 -= texSource.read(x - 1.0f * dx, y)[0] * 8.0f;
  t0 += texSource.read(x + 1.0f * dx, y)[0] * 8.0f;
  t0 -= texSource.read(x + 2.0f * dx, y)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x - 2.0f * dx, y)[0];
  t1 -= texTarget.read(x - 1.0f * dx, y)[0] * 8.0f;
  t1 += texTarget.read(x + 1.0f * dx, y)[0] * 8.0f;
  t1 -= texTarget.read(x + 2.0f * dx, y)[0];
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;

  // t derivative
  Iz[pos] = texTarget.read(x, y)[0] - texSource.read(x, y)[0];

  // y derivative
  t0 = texSource.read(x, y - 2.0f * dy)[0];
  t0 -= texSource.read(x, y - 1.0f * dy)[0] * 8.0f;
  t0 += texSource.read(x, y + 1.0f * dy)[0] * 8.0f;
  t0 -= texSource.read(x, y + 2.0f * dy)[0];
  t0 /= 12.0f;

  t1 = texTarget.read(x, y - 2.0f * dy)[0];
  t1 -= texTarget.read(x, y - 1.0f * dy)[0] * 8.0f;
  t1 += texTarget.read(x, y + 1.0f * dy)[0] * 8.0f;
  t1 -= texTarget.read(x, y + 2.0f * dy)[0];
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;
}

// void copy_image(dpct::image_accessor_ext<cl::sycl::float4, 2> acc, int *d, int cols, sycl::nd_item<3> item_ct1)
// {
//   cl::sycl::float4 tex_data = {0};
//   int idx = item_ct1.get_global_id(0);
  
//   tex_data[0] = d[idx];
  
//   int r = idx / cols;
//   int c = idx % cols;
//   acc.write(r,c) = tex_data;
// }

void test_image(dpct::image_accessor_ext<cl::sycl::float4, 2> s_acc, dpct::image_accessor_ext<cl::sycl::float4, 2> t_acc, const float *I0, const float *I1, int cols, sycl::nd_item<3> item_ct1) {
  
  cl::sycl::float4 tex_data0;
  cl::sycl::float4 tex_data1;
  
  int idx = item_ct1.get_global_id(0);
  
  tex_data0[0] = I0[idx];
  tex_data1[0] = I1[idx];
  
  int r = idx / cols;
  int c = idx % cols;
  
//   s_acc.read(r, c) = tex_data0;
//   t_acc.read(r, c) = tex_data1;
  tex_data0 = s_acc.read(r, c);
  tex_data1 = t_acc.read(r, c);
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
                               int s, float *Ix, float *Iy, float *Iz) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  dpct::image_wrapper_base_p texSource, texTarget;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I0);
  
  //198 - 201
//   int *d; size_t pitch;
//   d = (int *)dpct::dpct_malloc(pitch, w * sizeof(sycl::float4), h);

//   dpct::dpct_memcpy(d, s, h, w * sizeof(sycl::float4), w*sizeof(sycl::float4), h, dpct::host_to_device);
  
  /*
  DPCT1059:13: SYCL only supports 4-channel image format. Adjust the code.
  */
  texRes.set_channel(dpct::image_channel::create<sycl::float4>());
  //dpct::image_channel channel = dpct::image_channel::create<sycl::int4>();
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(float));
  //texRes.attach(d, w, h, s, channel);


  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::mirrored_repeat,
               sycl::filtering_mode::linear,
               sycl::coordinate_normalization_mode::normalized); // NOTE: Need to verufy with rak
  /*
  DPCT1004:14: Compatible DPC++ code could not be generated.
  */
  //texDescr.readMode = cudaReadModeElementType;
  texSource = dpct::create_image_wrapper(texRes, texDescr);
  
  memset(&texRes, 0, sizeof(dpct::image_data));
  
  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I1);
  

  /*
  DPCT1059:16: SYCL only supports 4-channel image format. Adjust the code.
  */
  texRes.set_channel(dpct::image_channel::create<sycl::float4>());        //+float4
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(float));
  texTarget = dpct::create_image_wrapper(texRes, texDescr);

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto acc_source =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texSource)->get_access(
            cgh);
    auto source_smpl = texSource->get_sampler();
    
     auto acc_target =
        static_cast<dpct::image_wrapper<cl::sycl::float4, 2> *>(texTarget)->get_access(
            cgh);
    auto target_smpl = texTarget->get_sampler();
    
    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
//      cgh.single_task<class dpct_single_kernel>(
//                      [=] {
          test_image(
              dpct::image_accessor_ext<cl::sycl::float4, 2>(source_smpl, acc_source), dpct::image_accessor_ext<cl::sycl::float4, 2>(target_smpl, acc_target), I0, I1, h, item_ct1);
          });
    
  }).wait();

  
  /*
  DPCT1049:12: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
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
  });
}
