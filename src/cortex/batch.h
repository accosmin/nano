#pragma once

#include "trainer_data.h"
#include "trainer_result.h"

namespace cortex
{
        class model_t;
        class criterion_t;

        ///
        /// \brief batch train the given model
        ///
        NANOCV_PUBLIC trainer_result_t batch_train(
                const model_t&, const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, const criterion_t& criterion,
                math::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                bool verbose = true);
}
