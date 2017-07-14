#pragma once

#include "trainer.h"

namespace nano
{
        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        struct batch_trainer_t final : public trainer_t
        {
                explicit batch_trainer_t(const string_t& params = string_t());

                virtual trainer_result_t train(
                        const enhancer_t&, const task_t&, const size_t fold, accumulator_t&) const override;
        };
}
