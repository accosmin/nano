#pragma once

#include "trainer.h"

namespace nano
{
        class stoch_optimizer_t;

        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(stochastic_trainer_t,
                        "parameters: opt=sg[...],epochs=16[1,1024],"\
                        "policy=stop_early[,all_epochs]")

                // constructor
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const override;

        private:

                trainer_result_t train(
                        const task_t&, const size_t fold,
                        const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0,
                        const stoch_optimizer_t&, const size_t epochs,
                        const trainer_policy, const bool verbose) const;
        };
}

