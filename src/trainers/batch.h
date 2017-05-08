#pragma once

#include "trainer.h"

namespace nano
{
        struct batch_optimizer_t;

        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        struct batch_trainer_t final : public trainer_t
        {
                explicit batch_trainer_t(const string_t& parameters = string_t());

                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads, const loss_t&, const sampler_t&,
                        model_t&) const override;
        };
}
