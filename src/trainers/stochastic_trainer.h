#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // stochastic trainer: the gradient update is computed on a random sample.
        //      the learning parameters are firstly tuned on a small subset and
        //      then used to optimize the loss on all samples.
        //
        // parameters:
        //      opt=asgd[,sgd]          - optimization method
        //      epoch=16[1,256]         - #epochs
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_state_t;

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "stochastic trainer, parameters: opt=asgd[,sgd],epoch=16[1,256]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // SGD algorithm (from a given state)
                void sgd(const task_t&, const samples_t&, const samples_t&, const loss_t&,
                         size_t iterations, size_t evalsize, stochastic_state_t& state) const;

        private:

                // attributes
                string_t                m_optimizer;
                size_t                  m_epochs;
        };
}

#endif // STOCHASTIC_TRAINER_H
