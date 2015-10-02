#pragma once

#include "cortex/trainer.h"

namespace ncv
{
        ///
        /// stochastic trainer: the gradient update is computed on a random sample with
        ///     a geometrically decreasing learning rate.
        ///
        /// parameters:
        ///     opt=sg[,sga,sia,nag,adagrad,adadelta]   - optimization method: SG, SGA, SIA, AG, AGGR, ADAGRAD, ADADELTA
        ///     epoch=16[1,1024]                        - #epochs (~ #samples)
        ///
        /// NB: "Minimizing Finite Sums with the Stochastic Average Gradient"
        ///     - Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(stochastic_trainer_t,
                                     "parameters: opt=sg[,sga,sia,ag,aggr,adagrad,adadelta],epoch=16[1,1024]")

                // constructor
                stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion, 
                        model_t&) const override;
        };
}

