#pragma once

#include "trainer.h"

namespace nano
{
        struct stoch_optimizer_t;

        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        struct stochastic_trainer_t final : public trainer_t
        {
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads, const loss_t&, const criterion_t&,
                        model_t&) const override;
        };
}

