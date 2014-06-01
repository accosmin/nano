#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        ///
        /// stochastic trainer: the gradient update is computed on a random sample with
        ///     a geometrically decreasing learning rate.
        ///
        /// parameters:
        ///     opt=sg[,sga,sia]        - optimization method: SG, SGA, SIA
        ///     epoch=16[1,1024]        - #epochs (~ #samples)
        ///
        /// NB: "Minimizing Finite Sums with the Stochastic Average Gradient"
        ///     - Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(stochastic_trainer_t)

                // constructor
                stochastic_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;
        };
}

#endif // STOCHASTIC_TRAINER_H
