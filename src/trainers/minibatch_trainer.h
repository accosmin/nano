#ifndef MINIBATCH_TRAINER_H
#define MINIBATCH_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // mini-batch trainer:
        //      the optimization is performed in epochs on small subsets of random samples,
        //              with the most promising partial models being further optimized.
        //
        // parameters:
        //      opt=cgd[,gd,lbfgs]      - optimization method / epoch
        //      eps=1e-6[1e-8,1e-3]     - convergence / epoch
        //      iters=16[4,256]         - maximum number of iterations / epoch
        //      batch=1024[256,16K]     - number of samples / epoch
        //      epoch=16[1,256]         - #epochs
        /////////////////////////////////////////////////////////////////////////////////////////

        class minibatch_trainer_t : public trainer_t
        {
        public:

                // constructor
                minibatch_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(minibatch_trainer_t, trainer_t,
                "minibatch trainer, parameters: opt=cgd[,gd,lbfgs],iters=16[4,256],eps=1e-6[1e-8,1e-3],batch=1024[256,16K],epoch=16[1,256]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // attributes
                string_t                m_optimizer;
                size_t                  m_iterations;
                scalar_t                m_epsilon;
                size_t                  m_batch;
                size_t                  m_epochs;
        };
}

#endif // MINIBATCH_TRAINER_H
