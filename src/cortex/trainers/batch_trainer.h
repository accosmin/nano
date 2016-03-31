#pragma once

#include "cortex/trainer.h"

namespace nano
{
        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        class batch_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(batch_trainer_t,
                        "parameters: opt=lbfgs[,cgd,gd],iters=1024[4,4096],eps=1e-4[1e-8,1e-3]")

                // constructor
                explicit batch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const override;
        };
}
