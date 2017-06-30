#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        struct stoch_trainer_t final : public trainer_t
        {
                explicit stoch_trainer_t(const string_t& params = string_t());

                virtual trainer_result_t train(
                        const iterator_t&, const task_t&, const size_t fold, const size_t nthreads, const loss_t&,
                        model_t&) const override;
        };
}
