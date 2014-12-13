#pragma once

#include "trainer.h"

namespace ncv
{
        ///
        /// mini-batch trainer: each gradient update is computed for a random sub-set of samples.
        ///
        /// parameters:
        ///     opt=gd[,lbfgs,cgd]      - optimization method
        ///     epoch=16[1,1024]        - #epochs (~ #samples)
        ///     batch=1024[32,8192]     - mini-batch size (#samples)
        ///     ratio=1.05[1.0,2.0]     - ratio to increase the batch size at each epoch
        ///     iters=8[4,128]          - maximum number of iterations
        ///     eps=1e-4[1e-8,1e-3]     - convergence
        ///
        class minibatch_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(minibatch_trainer_t,
                                     "parameters: opt=gd[,lbfgs,cgd],epoch=16[1,1024],"\
                                     "batch=1024[32,8192],ratio=1.05[1.0,2.0],iters=8[4,128],eps=1e-4[1e-8,1e-3]")

                // constructor
                minibatch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion, 
                        model_t&) const;
        };
}

