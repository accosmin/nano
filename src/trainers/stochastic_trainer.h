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
        //      gamma=0.01[1e-3,1e-1]   - starting learning rate
        //      beta=0.999[0.50,1.00]   - factor to geometrically decrease the learning rate
        //      batch=1024[256,16K]     - #samples to consider in one epoch
        //      epoch=16[1,256]         - #epochs
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                "stochastic trainer, parameters: gamma=0.01[1e-3,1e-1],beta=0.999[0.5,1.0],batch=1024[256,16K],epoch=16[1,256]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // attributes
                scalar_t                m_gamma;
                scalar_t                m_beta;
                size_t                  m_batch;
                size_t                  m_epochs;
        };
}

#endif // STOCHASTIC_TRAINER_H
