#pragma once

#include "trainer_data.h"
#include "trainer_result.h"

namespace ncv
{
        class model_t;

        ///
        /// \brief batch train the given model
        ///
        NANOCV_PUBLIC trainer_result_t batch_train(
                const model_t&, const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, const string_t& criterion,
                optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                bool verbose = true);
}
