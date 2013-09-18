#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // batch trainer: each gradient update is computed for all samples.
        //
        // parameters:
        //      opt=lbfgs[,cgd,gd]      - batch optimization method
        //      iter=256[4,4096]        - maximum number of iterations
        //      eps=1e-6[1e-8,1e-3]     - convergence
        /////////////////////////////////////////////////////////////////////////////////////////
                
//        - mini-batch size (100 - 10000)
//        - epoch size (#iterations with the same samples 4 - 64)
//        - number of opochs (8 - 1024)

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "batch trainer, parameters: opt=lbfgs[,cgd,gd],iter=256[4,4096],eps=1e-6[1e-8,1e-3]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_iterations;
                scalar_t        m_epsilon;
        };
}

#endif // STOCHASTIC_TRAINER_H
