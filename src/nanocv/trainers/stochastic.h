#pragma once

#include "trainer.h"

namespace ncv
{
        ///
        /// \brief stochastically train the given model
        ///
        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                stochastic_optimizer optimizer, size_t epochs);
}
