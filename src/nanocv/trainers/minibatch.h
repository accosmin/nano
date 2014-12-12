#pragma once

#include "trainer.h"

namespace ncv
{
        ///
        /// \brief minibatch train the given model
        ///
        trainer_result_t minibatch_train(
                const model_t&, const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, const string_t& criterion,
                batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon);
}
