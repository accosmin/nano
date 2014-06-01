#ifndef BATCH_TRAINER_H
#define BATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        ///
        /// batch trainer: each gradient update is computed for all samples.
        ///
        /// parameters:
        ///     opt=lbfgs[,cgd,gd]      - optimization method
        ///     iters=1024[4,4096]      - maximum number of iterations
        ///     eps=1e-6[1e-8,1e-3]     - convergence        
        ///
        class batch_trainer_t : public trainer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(batch_trainer_t)

                // constructor
                batch_trainer_t(const string_t& parameters = string_t());

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;
        };
}

#endif // BATCH_TRAINER_H
