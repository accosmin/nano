#pragma once

#include "optimizer.h"
#include "trainer_result.h"

namespace nano
{
        class loss_t;
        class task_t;
        struct fold_t;
        class model_t;
        class criterion_t;

        ///
        /// \brief minibatch train the given model
        ///
        NANO_PUBLIC trainer_result_t minibatch_train(
                const model_t&, const task_t&, const fold_t& tfold, const fold_t& vfold, const size_t nthreads,
                const loss_t&, const criterion_t& criterion,
                batch_optimizer optimizer, size_t epochs, scalar_t epsilon, bool verbose = true);
}
