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

#include "kvazaar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitstream.h"
#include "cfg.h"
#include "checkpoint.h"
#include "encoder.h"
#include "encoder_state-bitstream.h"
#include "encoder_state-ctors_dtors.h"
#include "encoderstate.h"
#include "global.h"
#include "image.h"
#include "input_frame_buffer.h"
#include "kvazaar_internal.h"
#include "ocl_helpers.h"
#include "strategyselector.h"
#include "threadqueue.h"
#include "videoframe.h"

// Forward declaration
static int get_opencl_stuff(encoder_control_t* encoder);

static void kvazaar_close(kvz_encoder *encoder)
{
  if (encoder) {
    if (encoder->states) {
      for (unsigned i = 0; i < encoder->num_encoder_states; ++i) {
        kvz_encoder_state_finalize(&encoder->states[i]);
      }
    }
    FREE_POINTER(encoder->states);
    kvz_encoder_control_free(encoder->control);
    encoder->control = NULL;
  }
  FREE_POINTER(encoder);
}


static kvz_encoder * kvazaar_open(const kvz_config *cfg)
{
  kvz_encoder *encoder = NULL;

  //Initialize strategies
  // TODO: Make strategies non-global
  if (!kvz_strategyselector_init(cfg->cpuid, KVZ_BIT_DEPTH)) {
    fprintf(stderr, "Failed to initialize strategies.\n");
    goto kvazaar_open_failure;
  }

  kvz_init_exp_golomb();

  encoder = calloc(1, sizeof(kvz_encoder));
  if (!encoder) {
    goto kvazaar_open_failure;
  }

  encoder->control = kvz_encoder_control_init(cfg);
  if (!encoder->control) {
    goto kvazaar_open_failure;
  }

  encoder->num_encoder_states = encoder->control->owf + 1;
  encoder->cur_state_num = 0;
  encoder->out_state_num = 0;
  encoder->frames_started = 0;
  encoder->frames_done = 0;

  kvz_init_input_frame_buffer(&encoder->input_buffer);

  // TODO: Check if opencl is enabled
  if (get_opencl_stuff(encoder->control)){
    goto kvazaar_open_failure;
  }

  encoder->states = calloc(encoder->num_encoder_states, sizeof(encoder_state_t));
  if (!encoder->states) {
    goto kvazaar_open_failure;
  }

  for (unsigned i = 0; i < encoder->num_encoder_states; ++i) {
    encoder->states[i].encoder_control = encoder->control;

    if (!kvz_encoder_state_init(&encoder->states[i], NULL)) {
      goto kvazaar_open_failure;
    }

    encoder->states[i].global->QP = (int8_t)cfg->qp;
  }

  for (int i = 0; i < encoder->num_encoder_states; ++i) {
    if (i == 0) {
      encoder->states[i].previous_encoder_state = &encoder->states[encoder->num_encoder_states - 1];
    } else {
      encoder->states[i].previous_encoder_state = &encoder->states[(i - 1) % encoder->num_encoder_states];
    }
    kvz_encoder_state_match_children_of_previous_frame(&encoder->states[i]);
  }

  encoder->states[encoder->cur_state_num].global->frame = -1;

  return encoder;

kvazaar_open_failure:
  kvazaar_close(encoder);
  return NULL;
}


static void set_frame_info(kvz_frame_info *const info, const encoder_state_t *const state)
{
  info->poc = state->global->poc,
  info->qp = state->global->QP;
  info->nal_unit_type = state->global->pictype;
  info->slice_type = state->global->slicetype;
  kvz_encoder_get_ref_lists(state, info->ref_list_len, info->ref_list);
}


static int kvazaar_headers(kvz_encoder *enc,
                           kvz_data_chunk **data_out,
                           uint32_t *len_out)
{
  if (data_out) *data_out = NULL;
  if (len_out) *len_out = 0;

  bitstream_t stream;
  kvz_bitstream_init(&stream);

  kvz_encoder_state_write_parameter_sets(&stream, &enc->states[enc->cur_state_num]);

  // Get stream length before taking chunks since that clears the stream.
  if (len_out) *len_out = kvz_bitstream_tell(&stream) / 8;
  if (data_out) *data_out = kvz_bitstream_take_chunks(&stream);

  kvz_bitstream_finalize(&stream);
  return 1;
}


/**
* \brief Separate a single field from a frame.
*
* \param frame_in           input frame to extract field from
* \param source_scan_type   scan type of input material (0: progressive, 1:top field first, 2:bottom field first)
* \param field parity   
* \param field_out
*
* \return              1 on success, 0 on failure
*/
static int yuv_io_extract_field(const kvz_picture *frame_in, unsigned source_scan_type, unsigned field_parity, kvz_picture *field_out)
{
  if ((source_scan_type != 1) && (source_scan_type != 2)) return 0;
  if ((field_parity != 0)     && (field_parity != 1))     return 0;

  unsigned offset = 0;
  if (source_scan_type == 1) offset = field_parity ? 1 : 0;
  else if (source_scan_type == 2) offset = field_parity ? 0 : 1;  

  //Luma
  for (int i = 0; i < field_out->height; ++i){
    kvz_pixel *row_in  = frame_in->y + MIN(frame_in->height - 1, 2 * i + offset) * frame_in->stride;
    kvz_pixel *row_out = field_out->y + i * field_out->stride;
    memcpy(row_out, row_in, sizeof(kvz_pixel) * frame_in->width);
  }

  //Chroma
  for (int i = 0; i < field_out->height / 2; ++i){
    kvz_pixel *row_in = frame_in->u + MIN(frame_in->height / 2 - 1, 2 * i + offset) * frame_in->stride / 2;
    kvz_pixel *row_out = field_out->u + i * field_out->stride / 2;
    memcpy(row_out, row_in, sizeof(kvz_pixel) * frame_in->width / 2);
  }

  for (int i = 0; i < field_out->height / 2; ++i){
    kvz_pixel *row_in = frame_in->v + MIN(frame_in->height / 2 - 1, 2 * i + offset) * frame_in->stride / 2;
    kvz_pixel *row_out = field_out->v + i * field_out->stride / 2;
    memcpy(row_out, row_in, sizeof(kvz_pixel) * frame_in->width / 2);
  }

  return 1;
}


static int kvazaar_encode(kvz_encoder *enc,
                          kvz_picture *pic_in,
                          kvz_data_chunk **data_out,
                          uint32_t *len_out,
                          kvz_picture **pic_out,
                          kvz_picture **src_out,
                          kvz_frame_info *info_out)
{
  if (data_out) *data_out = NULL;
  if (len_out) *len_out = 0;
  if (pic_out) *pic_out = NULL;
  if (src_out) *src_out = NULL;

  encoder_state_t *state = &enc->states[enc->cur_state_num];

  if (!state->prepared) {
    kvz_encoder_next_frame(state);
  }

  if (pic_in != NULL) {
    // FIXME: The frame number printed here is wrong when GOP is enabled.
    CHECKPOINT_MARK("read source frame: %d", state->global->frame + enc->control->cfg->seek);
  }

  if (kvz_encoder_feed_frame(&enc->input_buffer, state, pic_in)) {
    assert(state->global->frame == enc->frames_started);
    // Start encoding.
    ocl_pre_calculate_mvs(state);
    kvz_encode_one_frame(state);
    enc->frames_started += 1;
  }

  // If we have finished encoding as many frames as we have started, we are done.
  if (enc->frames_done == enc->frames_started) {
    return 1;
  }

  if (!state->frame_done) {
    // We started encoding a frame; move to the next encoder state.
    enc->cur_state_num = (enc->cur_state_num + 1) % (enc->num_encoder_states);
  }

  encoder_state_t *output_state = &enc->states[enc->out_state_num];
  if (!output_state->frame_done &&
      (pic_in == NULL || enc->cur_state_num == enc->out_state_num)) {

    kvz_threadqueue_waitfor(enc->control->threadqueue, output_state->tqj_bitstream_written);
    // The job pointer must be set to NULL here since it won't be usable after
    // the next frame is done.
    output_state->tqj_bitstream_written = NULL;

    // Get stream length before taking chunks since that clears the stream.
    if (len_out) *len_out = kvz_bitstream_tell(&output_state->stream) / 8;
    if (data_out) *data_out = kvz_bitstream_take_chunks(&output_state->stream);
    if (pic_out) *pic_out = kvz_image_copy_ref(output_state->tile->frame->rec);
    if (src_out) *src_out = kvz_image_copy_ref(output_state->tile->frame->source);
    if (info_out) set_frame_info(info_out, output_state);

    output_state->frame_done = 1;
    output_state->prepared = 0;
    enc->frames_done += 1;

    enc->out_state_num = (enc->out_state_num + 1) % (enc->num_encoder_states);
  }

  return 1;
}


static int kvazaar_field_encoding_adapter(kvz_encoder *enc,
                                          kvz_picture *pic_in,
                                          kvz_data_chunk **data_out,
                                          uint32_t *len_out,
                                          kvz_picture **pic_out,
                                          kvz_picture **src_out,
                                          kvz_frame_info *info_out)
{
  if (enc->control->cfg->source_scan_type == KVZ_INTERLACING_NONE) {
    // For progressive, simply call the normal encoding function.
    return kvazaar_encode(enc, pic_in, data_out, len_out, pic_out, src_out, info_out);
  }

  // For interlaced, make two fields out of the input frame and call encode on them separately.
  encoder_state_t *state = &enc->states[enc->cur_state_num];
  kvz_picture *first_field = NULL, *second_field = NULL;
  struct {
    kvz_data_chunk* data_out;
    uint32_t len_out;
  } first = { 0 }, second = { 0 };

  if (pic_in != NULL) {
    first_field = kvz_image_alloc(state->encoder_control->in.width, state->encoder_control->in.height);
    if (first_field == NULL) {
      goto kvazaar_field_encoding_adapter_failure;
    }
    second_field = kvz_image_alloc(state->encoder_control->in.width, state->encoder_control->in.height);
    if (second_field == NULL) {
      goto kvazaar_field_encoding_adapter_failure;
    }

    yuv_io_extract_field(pic_in, pic_in->interlacing, 0, first_field);
    yuv_io_extract_field(pic_in, pic_in->interlacing, 1, second_field);
    
    first_field->pts = pic_in->pts;
    first_field->dts = pic_in->dts;
    first_field->interlacing = pic_in->interlacing;

    // Should the second field have higher pts and dts? It shouldn't affect anything.
    second_field->pts = pic_in->pts;
    second_field->dts = pic_in->dts;
    second_field->interlacing = pic_in->interlacing;
  }

  if (!kvazaar_encode(enc, first_field, &first.data_out, &first.len_out, pic_out, NULL, info_out)) {
    goto kvazaar_field_encoding_adapter_failure;
  }
  if (!kvazaar_encode(enc, second_field, &second.data_out, &second.len_out, NULL, NULL, NULL)) {
    goto kvazaar_field_encoding_adapter_failure;
  }

  kvz_image_free(first_field);
  kvz_image_free(second_field);

  // Concatenate bitstreams.
  if (len_out != NULL) {
    *len_out = first.len_out + second.len_out;
  }
  if (data_out != NULL) {
    *data_out = first.data_out;
    if (first.data_out != NULL) {
      kvz_data_chunk *chunk = first.data_out;
      while (chunk->next != NULL) {
        chunk = chunk->next;
      }
      chunk->next = second.data_out;
    }
  }

  if (src_out != NULL) {
    // TODO: deinterlace the fields to one picture.
  }

  return 1;

kvazaar_field_encoding_adapter_failure:
  kvz_image_free(first_field);
  kvz_image_free(second_field);
  kvz_bitstream_free_chunks(first.data_out);
  kvz_bitstream_free_chunks(second.data_out);
  return 0;
}

static int get_opencl_stuff(const encoder_control_t* const encoder)
{
  // TODO: Have user select used (GPU) device
  int err = CL_SUCCESS;
  cl_uint numPlat;
  cl_device_id device_id;
  FILE *program_handle;
  size_t program_size;
  char *program_buffer;

 cl_context* context = encoder->opencl_structs.mve_fullsearch_context;
 cl_command_queue* commands = encoder->opencl_structs.mve_fullsearch_cqueue;
 cl_program* program = encoder->opencl_structs.mve_fullsearch_prog;

  err = clGetPlatformIDs(0 , NULL , &numPlat);
  if (err != CL_SUCCESS) return err;

  cl_platform_id* platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlat);
  err = clGetPlatformIDs(numPlat , platform , NULL);
  if (err != CL_SUCCESS) return err;

  for (unsigned int i = 0; i < numPlat; i++) {
    err = clGetDeviceIDs(platform[i] , CL_DEVICE_TYPE_GPU , 1 , &device_id , NULL);
    if (err == CL_SUCCESS && i != 2) break;
  }
  free(platform);
  if (err != CL_SUCCESS) return err;
  *context = clCreateContext(0 , 1 , &device_id , NULL , NULL , &err);
  if (err != CL_SUCCESS) return err;

  *commands = clCreateCommandQueue(*context , device_id , CL_QUEUE_PROFILING_ENABLE , &err);
  if (err != CL_SUCCESS) return err;

  program_handle = fopen("fullsearch_kernel.cl" , "rb");
  if (program_handle == NULL) return 1;

  fseek(program_handle , 0 , SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer , sizeof(char) , program_size , program_handle);
  fclose(program_handle);
  *program = clCreateProgramWithSource(*context , 1 , &program_buffer , 0 , &err);
  free(program_buffer);
  if (err != CL_SUCCESS) return err;

  //TODO: Check if it's possible to allocate the sad buffer on device
  int search_range = 32;
  switch (encoder->cfg->ime_algorithm) {
  case KVZ_IME_FULL64: search_range = 64; break;
  case KVZ_IME_FULL32: search_range = 32; break;
  case KVZ_IME_FULL16: search_range = 16; break;
  case KVZ_IME_FULL8: search_range = 8; break;
  default: break;
  }
  char build_opts[64];
  sprintf(build_opts , "-D SEARCH_RANGE=%d -D WIDTH=%d -D HEIGHT=%d" , search_range , encoder->in.width , encoder->in.height);

  err = clBuildProgram(*program , 1 , &device_id , build_opts , NULL , NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    unsigned char buffer[2048];

    clGetProgramBuildInfo(*program , device_id , CL_PROGRAM_BUILD_LOG , sizeof(buffer) , buffer , &len);
    printf("%p" , buffer);
    return err;
  }
  return 0;
}

static const kvz_api kvz_8bit_api = {
  .config_alloc = kvz_config_alloc,
  .config_init = kvz_config_init,
  .config_destroy = kvz_config_destroy,
  .config_parse = kvz_config_parse,

  .picture_alloc = kvz_image_alloc,
  .picture_free = kvz_image_free,

  .chunk_free = kvz_bitstream_free_chunks,

  .encoder_open = kvazaar_open,
  .encoder_close = kvazaar_close,
  .encoder_headers = kvazaar_headers,
  .encoder_encode = kvazaar_field_encoding_adapter,
};


const kvz_api * kvz_api_get(int bit_depth)
{
  return &kvz_8bit_api;
}

