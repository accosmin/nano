#ifndef LAYERWISE_TRAINER_H
#define LAYERWISE_TRAINER_H

#include "trainer.h"

namespace ncv
{
        ///
        /// layer-wise trainer: a single layer is trained at a time (in a forward network)
        ///     using a mini-batch approach.
        ///
        /// parameters:
        ///     opt=gd[,lbfgs,cgd]      - optimization method
        ///     epoch=16[1,1024]        - #epochs (~ #samples)
        ///     batch=1024[256,8192]    - mini-batch size (#samples)
        ///     iters=8[4,128]          - maximum number of iterations
        ///     eps=1e-6[1e-8,1e-3]     - convergence
        ///
        class layerwise_trainer_t : public trainer_t
        {
        public:

                // constructor
                layerwise_trainer_t(const string_t& parameters = string_t());

                // create an object clone
                virtual rtrainer_t clone(const string_t& parameters) const
                {
                        return rtrainer_t(new layerwise_trainer_t(parameters));
                }

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;
        };
}

#endif // LAYERWISE_TRAINER_H
