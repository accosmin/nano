#pragma once

#include "trainer.h"
#include "optim/batch/types.h"

namespace nano
{
        ///
        /// \brief batch trainer: each gradient update is computed for all samples.
        ///
        class batch_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(batch_trainer_t,
                        "parameters: opt=lbfgs[...],epochs=1024[4,4096],policy=stop_early[,all_epochs]")

                // constructor
                explicit batch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const override;

        private:

                trainer_result_t train(
                        const task_t&, const size_t fold,
                        const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0,
                        const batch_optimizer, const size_t epochs, const scalar_t epsilon,
                        const trainer_policy, const bool verbose) const;
        };
}
