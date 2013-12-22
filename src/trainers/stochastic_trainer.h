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
        //      epoch=4[2,16]              - optimization iterations (~ #samples)
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "stochastic trainer, parameters: epoch=4[2,16]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // utility functions: update the searching grid for the optimum parameters
                static void update_range(scalar_t& min, scalar_t& max, scalar_t opt, scalar_t delta)
                {
                        max = opt + delta;
                        min = opt - delta;
                }

                // SGD algorithm
                scalar_t sgd(const task_t&, const samples_t&, const loss_t&, model_t&, vector_t& x,
                             scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const;

                // tune SGD learning parameters
                void tune_sgd(const task_t&, const samples_t&, const loss_t&, size_t nthreads, model_t&, vector_t&,
                              scalar_t& opt_log_gamma, scalar_t& opt_log_lambda) const;

        private:

                // attributes
                size_t                  m_epoch;
        };
}

#endif // STOCHASTIC_TRAINER_H
