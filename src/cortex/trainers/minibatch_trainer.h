#pragma once

#include "cortex/trainer.h"

namespace nano
{
        ///
        /// \brief mini-batch trainer: each gradient update is computed for a random sub-set of samples.
        ///
        class minibatch_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(minibatch_trainer_t,
                        "parameters: opt=cgd[,lbfgs,gd],epoch=16[1,1024],eps=1e-6[1e-8,1e-3]")

                // constructor
                explicit minibatch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const override;
        };
}

