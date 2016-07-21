#ifndef OCL_HELPERS
#define OCL_HELPERS
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

#include <CL\cl.h>

#include "encoderstate.h"
#include "global.h"

void ocl_expand_rec(encoder_state_t* state , kvz_picture* im);

int ocl_pre_calculate_mvs(encoder_state_t* state);

mv_buffers* ocl_mv_buffers_alloc();

size_t ocl_return_index(const encoder_state_t* const state , int width , int height , const vector2d_t* const orig);

#endif //OCL_HELPERS
