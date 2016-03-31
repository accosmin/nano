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
        /// \brief batch train the given model
        ///
        NANO_PUBLIC trainer_result_t batch_train(
                const model_t&, const task_t&, const size_t fold, const size_t nthreads,
                const loss_t&, const criterion_t& criterion,
                const batch_optimizer optimizer, const size_t iterations, const scalar_t epsilon,
                bool verbose = true);
}
