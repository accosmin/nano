#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        ///
        /// stochastic trainer: the gradient update is computed on a random sample with
        ///      a geometrically decreasing learning rate.
        ///
        /// parameters:
        ///      opt=sgd[,asgd]          - optimization method: (average) stochastic gradient descent
        ///      epoch=16[1,1024]        - #epochs (~ #samples)
        ///
        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                "stochastic trainer, parameters: opt=sgd[,asgd],epoch=16[1,1024]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // attributes
                string_t                m_optimizer;
                size_t                  m_epochs;
        };
}

#endif // STOCHASTIC_TRAINER_H
