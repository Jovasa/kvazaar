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

  cl_kernel expand = clCreateKernel(*state->encoder_control->opencl_structs.mve_fullsearch_prog , "expand" , NULL);
  cl_mem input_buffer = clCreateBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_context , CL_MEM_READ_ONLY , im->width*im->height*sizeof(kvz_pixel) , NULL , NULL);
  im->exp_luma_buffer = clCreateBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_context , CL_MEM_READ_ONLY , 
    (im->width+(search_range<<2))*(im->height+(search_range<<2))*sizeof(kvz_pixel) , NULL , NULL);

  clEnqueueWriteBuffer(*state->encoder_control->opencl_structs.mve_fullsearch_cqueue , input_buffer , CL_FALSE ,
    0 , im->width*im->height*sizeof(kvz_pixel) , im->y , 0 , NULL , &write_ready);

  clSetKernelArg(expand , 0 , sizeof(cl_mem) , &input_buffer);
  clSetKernelArg(expand , 1 , sizeof(cl_mem) , &im->exp_luma_buffer);

  const size_t expand_image_size[2] = {im->width+(search_range<<2), im->height+(search_range<<2)};
  const size_t work_group[2] = {16 , 16};

  clEnqueueNDRangeKernel(*state->encoder_control->opencl_structs.mve_fullsearch_cqueue , expand , 2 , 0 , expand_image_size , work_group , 1 , &write_ready , im->expand_ready);
  clReleaseMemObject(input_buffer);
  clReleaseKernel(expand);
}