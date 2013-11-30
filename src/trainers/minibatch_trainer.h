#ifndef MINIBATCH_TRAINER_H
#define MINIBATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // mini-batch trainer: the gradient update is computed
        //      on a random mini-batch of fixed size in epochs.
        //
        // parameters:
        //      opt=lbfgs[,cgd,gd]              - optimization method
        //      iters=4[1,128]                  - maximum number of optimization iterations per epoch
        //      batch=1024[100,10000]           - mini-batch size / epoch
        //      epoch=256[8,1024]               - number of epochs
        /////////////////////////////////////////////////////////////////////////////////////////

        class minibatch_trainer_t : public trainer_t
        {
        public:

                // constructor
                minibatch_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(minibatch_trainer_t, trainer_t,
                                  "mini-batch trainer, parameters: opt=lbfgs[,cgd,gd],iters=4[1,128],batch=1024[100,10000],epoch=256[8,1024]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // attributes
                string_t        m_optimizer;
                size_t          m_iterations;
                size_t          m_batchsize;
                size_t          m_epochs;
                scalar_t        m_epsilon;
        };
}

#endif // MINIBATCH_TRAINER_H
