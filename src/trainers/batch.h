#pragma once

#include "trainer.h"

namespace nano
{
        class batch_optimizer_t;

        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        class batch_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(batch_trainer_t)

                // constructor
                explicit batch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads, const loss_t&, const criterion_t&,
                        model_t&) const override final;
        };
}
