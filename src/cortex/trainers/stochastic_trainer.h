#pragma once

#include "cortex/trainer.h"

namespace nano
{
        ///
        /// \brief stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(stochastic_trainer_t,
                        "parameters: opt=sg[,sng,sgm,ag,agfr,aggr,adagrad,adadelta,adam],epochs=16[1,1024]")

                // constructor
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const size_t fold, const size_t nthreads,
                        const loss_t&, const criterion_t& criterion,
                        model_t&) const override;
        };
}

