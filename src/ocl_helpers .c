/*****************************************************************************
* This file is part of Kvazaar HEVC encoder.
*
* Copyright (C) 2013-2015 Tampere University of Technology and others (see
* COPYING file).
*
* Kvazaar is free software: you can redistribute it and/or modify it under
* the terms of the GNU Lesser General Public License as published by the
* Free Software Foundation; either version 2.1 of the License, or (at your
* option) any later version.
*
* Kvazaar is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
* more details.
*
* You should have received a copy of the GNU General Public License along
* with Kvazaar.  If not, see <http://www.gnu.org/licenses/>.
****************************************************************************/

#include "ocl_helpers.h"

static size_t * returnBufferSizes(int width, int height)
{
  static size_t sizes[17];
  for (int i = 0; i < 3; i++) {
    sizes[i] = sizeof(cl_int2) * width / (8 << i)*height / (8 << i);
  }
  for (int i = 3; i < 9; i++) {
    sizes[i] = sizes[i % 3] / 2;
  }
  for (int i = 9; i < 17; i++) {
    sizes[i] = sizes[4 + (i + 1) % 2];
  }
  return sizes;
}

void ocl_expand_rec(encoder_state_t* state , kvz_picture* im)
{
  cl_event write_ready;
  im->expand_ready = MALLOC(cl_event , 1);
  int search_range = 32;
  switch (state->encoder_control->cfg->ime_algorithm) {
  case KVZ_IME_FULL64: search_range = 64; break;
  case KVZ_IME_FULL32: search_range = 32; break;
  case KVZ_IME_FULL16: search_range = 16; break;
  case KVZ_IME_FULL8: search_range = 8; break;
  default: break;
  }

  cl_mem input_buffer = clCreateBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_context , CL_MEM_READ_ONLY ,
    im->width*im->height*sizeof(kvz_pixel) , NULL , NULL);
  im->exp_luma_buffer = clCreateBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_context , CL_MEM_READ_ONLY , 
    (im->width+(search_range<<2))*(im->height+(search_range<<2))*sizeof(kvz_pixel) , NULL , NULL);

  clEnqueueWriteBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_cqueue , input_buffer , CL_FALSE ,
    0 , im->width*im->height*sizeof(kvz_pixel) , im->y , 0 , NULL , &write_ready);

  clSetKernelArg(state->kernels.expand_kernel , 0 , sizeof(cl_mem) , &input_buffer);
  clSetKernelArg(state->kernels.expand_kernel , 1 , sizeof(cl_mem) , &im->exp_luma_buffer);

  const size_t expand_image_size[2] = {im->width+(search_range<<2), im->height+(search_range<<2)};
  const size_t work_group[2] = {16 , 16};

  clEnqueueNDRangeKernel(*state->encoder_control->opencl_structs.mve_fullsearch_cqueue , state->kernels.expand_kernel ,
    2 , 0 , expand_image_size , work_group , 1 , &write_ready , im->expand_ready);
  clReleaseMemObject(input_buffer);
}

int ocl_pre_calculate_mvs(struct encoder_state_t* state)
{
  int err = CL_SUCCESS;
  // Values that are needed for calculating the mvs
  int buffers_used = 4;
  int width = state->encoder_control->cfg->width;
  int height = state->encoder_control->cfg->height;
  size_t * sizes = returnBufferSizes(width , height);
  int search_range = 32;
  const size_t workgroup_size[3] = {1 , 1 , 64};
  const size_t sad_calc_kernel_size[3] = {width>>3, height>>3, 64};
  switch (state->encoder_control->cfg->ime_algorithm) {
  case KVZ_IME_FULL64: search_range = 64; break;
  case KVZ_IME_FULL32: search_range = 32; break;
  case KVZ_IME_FULL16: search_range = 16; break;
  case KVZ_IME_FULL8: search_range = 8; break;
  default: break;
  }
  buffers_used += state->encoder_control->cfg->smp_enable ? 6 : 0;
  buffers_used += state->encoder_control->cfg->amp_enable ? 8 : 0;

  cl_kernel* calc_sads = &state->kernels.calc_sad_kernel;
  cl_kernel* reuse_sads = &state->kernels.reuse_sad_kernel;
  cl_kernel* reuse_sads_amp = &state->kernels.reuse_sad_kernel_amp;
  {
    // Set arguments for calc sads kernel that stay the same regardless, 
    size_t local = (search_range * 2 + 1)*(search_range * 2 + 1)*sizeof(kvz_pixel);
    err = clSetKernelArg(*calc_sads , 0 , sizeof(cl_mem) , &state->tile->frame->source->exp_luma_buffer);
    err = clSetKernelArg(*calc_sads , 3 , local , NULL);
  }

  for (int refs_used = 0; state->global->ref->used_size != refs_used; refs_used++) {
    cl_event sads_ready;
    state->global->buffers[refs_used] = malloc(sizeof(mv_buffers));
    mv_buffers* bufs = state->global->buffers[refs_used];
    bufs->buffers = malloc(sizeof(cl_mem*)*buffers_used);
    bufs->ready = malloc(sizeof(cl_event*)*buffers_used);

    cl_mem sad_buffer = clCreateBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_context , CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS ,
      sizeof(cl_short)*(width>>3)*(height>>3)*((search_range * 2 + 1)*(search_range * 2 + 1) + 1) , NULL , &err);

    // Set the arguments that are changing depending on the reference
    err = clSetKernelArg(*calc_sads , 1 , sizeof(cl_mem) , &state->global->ref->images[refs_used]->exp_luma_buffer);
    err = clSetKernelArg(*calc_sads , 2 , sizeof(cl_mem) , &sad_buffer);

    err = clEnqueueNDRangeKernel(*state->encoder_control->opencl_structs.mve_fullsearch_cqueue , *calc_sads , 3 , NULL , sad_calc_kernel_size , workgroup_size , 1 ,
      state->global->ref->images[refs_used]->expand_ready, &sads_ready);

    for (int i = 1; i != buffers_used; i++) {
      break;
    }
    clReleaseMemObject(sad_buffer);
    clReleaseEvent(sads_ready);
  }
  return 1;
}