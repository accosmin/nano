#pragma once

#include "trainer_data.h"
#include "trainer_result.h"

namespace nano
{
        class model_t;
        class criterion_t;

        ///
        /// \brief stochastically train the given model
        ///
        NANO_PUBLIC trainer_result_t stochastic_train(
                const model_t&, const task_t&, const fold_t& tfold, const fold_t& vfold, const size_t nthreads,
                const loss_t&, const criterion_t&,
                nano::stoch_optimizer optimizer, size_t epochs,
                bool verbose = true);
}
