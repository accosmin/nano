#pragma once

#include "cortex/trainer.h"

namespace cortex
{
        ///
        /// batch trainer: each gradient update is computed for all samples.
        ///
        /// parameters:
        ///     opt=lbfgs[,cgd,gd]              - optimization method
        ///     iters=1024[4,4096]              - maximum number of iterations
        ///     eps=1e-4[1e-8,1e-3]             - convergence
        ///
        class batch_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(batch_trainer_t,
                                     "parameters: opt=lbfgs[,cgd,gd],iters=1024[4,4096],eps=1e-4[1e-8,1e-3]")

                // constructor
                explicit batch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion, 
                        model_t&) const override;
        };
}
