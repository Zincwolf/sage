#pragma once

#include <torch/extension.h>

torch::Tensor sph_gemv_forward_cuda(torch::Tensor input,
                                    torch::Tensor packed_codes,
                                    torch::Tensor radius_extra,
                                    torch::Tensor radius_scales,
                                    torch::Tensor radius_mins,
                                    torch::Tensor codebook,
                                    int64_t group_size);

torch::Tensor sphq_gemv_forward_cuda(torch::Tensor input,
                                   torch::Tensor packed_codes,
                                   torch::Tensor radius_extra,
                                   torch::Tensor radius_scales,
                                   torch::Tensor radius_mins,
                                   torch::Tensor codebook,
                                   int64_t group_size);
