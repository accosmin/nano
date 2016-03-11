#pragma once

#include "cortex/trainer.h"

namespace nano
{
        ///
        /// stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        /// parameters:
        ///     opt=sg[,...]            - optimization method: SG, SGM, AG, AGFR, AGGR, ADAGRAD, ADADELTA, ADAM
        ///     epoch=16[1,1024]        - #epochs (~ #samples)
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANO_MAKE_CLONABLE(stochastic_trainer_t,
                                     "parameters: opt=sg[,sgm,ag,agfr,aggr,adagrad,adadelta,adam],epoch=16[1,1024]")

                // constructor
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const criterion_t&,
                        model_t&) const override;
        };
}

