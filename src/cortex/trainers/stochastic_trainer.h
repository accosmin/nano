#pragma once

#include "cortex/trainer.h"

namespace cortex
{
        ///
        /// stochastic trainer: each gradient update is computed for a random sub-set of samples.
        ///
        /// parameters:
        ///     opt=sg[,...]            - optimization method: SG, SGA, SGM, SIA, AG, AGFR, AGGR, ADAGRAD, ADADELTA
        ///     epoch=16[1,1024]        - #epochs (~ #samples)
        ///
        /// NB: "Minimizing Finite Sums with the Stochastic Average Gradient"
        ///     - Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(stochastic_trainer_t,
                                     "parameters: opt=sg[,sga,sgm,sia,ag,agfr,aggr,adagrad,adadelta],epoch=16[1,1024]")

                // constructor
                explicit stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const criterion_t&,
                        model_t&) const override;
        };
}

