#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // stochastic trainer: each gradient update is computed for all samples.
        //
        // parameters:
        //      opt=asgd[,sgd]          - stochastic optimization method
        //      epochs=4[1,32]          - maximum number of iterations
        //      eps=1e-6[1e-3,1e-8]     - convergence
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "stochastic trainer, parameters: opt=asgd[,sgd],epochs=4[1,32],eps=1e-6[1e-3,1e-8]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_epochs;
                scalar_t        m_epsilon;
        };
}

#endif // STOCHASTIC_TRAINER_H
