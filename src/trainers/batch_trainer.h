#ifndef BATCH_TRAINER_H
#define BATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // batch trainer: each gradient update is computed for all samples.
        //
        // parameters:
        //      opt=lbfgs[,cgd,gd]      - optimization method
        //      eps=1e-6[1e-8,1e-3]     - convergence
        //      iters=1024[4,4096]      - maximum number of iterations
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class batch_trainer_t : public trainer_t
        {
        public:

                // constructor
                batch_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(batch_trainer_t, trainer_t,
                                  "batch trainer, parameters: opt=lbfgs[,cgd,gd],iters=1024[4,4096],eps=1e-6[1e-8,1e-3]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_iterations;
                scalar_t        m_epsilon;
        };
}

#endif // BATCH_TRAINER_H
