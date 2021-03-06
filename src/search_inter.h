#ifndef SEARCH_INTER_H_
#define SEARCH_INTER_H_
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

/**
 * \ingroup Compression
 * \file
 * Inter prediction parameter search.
 */

#include "cu.h"
#include "encoderstate.h"
#include "global.h" // IWYU pragma: keep
#include "inter.h"

typedef int kvz_mvd_cost_func(encoder_state_t * const state,
                              int x, int y,
                              int mv_shift,
                              int16_t mv_cand[2][2],
                              inter_merge_cand_t merge_cand[MRG_MAX_NUM_CANDS],
                              int16_t num_cand,
                              int32_t ref_idx,
                              uint32_t *bitcost);

void kvz_search_cu_inter(encoder_state_t * const state,
                         int x, int y, int depth,
                         lcu_t *lcu,
                         double *inter_cost,
                         uint32_t *inter_bitcost);

void kvz_search_cu_smp(encoder_state_t * const state,
                       int x, int y,
                       int depth,
                       part_mode_t part_mode,
                       lcu_t *lcu,
                       double *inter_cost,
                       uint32_t *inter_bitcost);

#endif // SEARCH_INTER_H_
