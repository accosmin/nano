#pragma once

#include "../trainer.h"

namespace ncv
{
        ///
        /// mini-batch trainer: each gradient update is computed for a random sub-set of samples.
        ///
        /// parameters:
        ///     opt=gd[,lbfgs,cgd]              - optimization method
        ///     epoch=16[1,1024]                - #epochs (~ #samples)
        ///     eps=1e-4[1e-8,1e-3]             - convergence
        ///
        class minibatch_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(minibatch_trainer_t,
                                     "parameters: opt=gd[,lbfgs,cgd],epoch=16[1,1024],eps=1e-4[1e-8,1e-3]")

                // constructor
                minibatch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion,
                        model_t&) const override;
        };
}

