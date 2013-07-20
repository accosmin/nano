#ifndef BATCH_TRAINER_H
#define BATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // batch trainer: each gradient update is computed for all samples.
        //
        // parameters:
        //      opt=lbfgs[,cgd,gd]      - batch optimization method
        //      iter=256[16,1024]       - maximum number of iterations
        //      eps=1e-5[1e-3,1e-6]     - convergence
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class batch_trainer_t : public trainer_t
        {
        public:

                // constructor
                batch_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(batch_trainer_t, trainer_t,
                                  "batch trainer, parameters: opt=lbfgs[,cgd,gd],iter=256[16,1024],eps=1e-5[1e-3,1e-6]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_iterations;
                scalar_t        m_epsilon;
        };
}

#endif // BATCH_TRAINER_H
