#pragma once

#include "trainer_data.h"
#include "trainer_result.h"

namespace cortex
{
        class model_t;

        ///
        /// \brief stochastically train the given model
        ///
        NANOCV_PUBLIC trainer_result_t stochastic_train(
                const model_t&, const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, const string_t& criterion,
                math::stoch_optimizer optimizer, size_t epochs,
                bool verbose = true);
}
