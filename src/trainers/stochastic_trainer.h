#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // stochastic trainer: the gradient update is computed on a random sample with
        //      a geometrically decreasing learning rate.
        //
        // parameters:
        //      opt=sgd[,asgd]          - optimization method: (average) stochastic gradient descent
        //      alpha=1e-2[1e-6,1e-1]   - starting learning rate
        //      epoch=16[1,256]          - #epochs (~ #samples)
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                "stochastic trainer, parameters: opt=sgd[,asgd],alpha=1e-3[1e-6,1e-1],epoch=16[1,256]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // attributes
                string_t                m_optimizer;
                scalar_t                m_alpha;
                size_t                  m_epochs;
        };
}

#endif // STOCHASTIC_TRAINER_H
