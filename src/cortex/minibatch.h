#pragma once

#include "trainer_data.h"
#include "trainer_result.h"

namespace zob
{
        class model_t;
        class criterion_t;

        ///
        /// \brief minibatch train the given model
        ///
        ZOB_PUBLIC trainer_result_t minibatch_train(
                const model_t&, const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, const criterion_t& criterion,
                zob::batch_optimizer optimizer, size_t epochs, scalar_t epsilon, bool verbose = true);
}
