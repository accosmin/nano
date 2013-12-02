#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // stochastic trainer: the gradient update is computed on a random sample.
        //      the parameters are tuned in epochs of <batch> fixed size.
        //
        // parameters:
        //      iters=1024[100,10000]      - number of iterations per tuning depth
        //      depth=4[2,16]              - tunning depth (accuracy)
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "stochastic trainer, parameters: iters=1024[100,10000],depth=4[2,16]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const;

        private:

                // utility functions: update the searching grid for the optimum parameters
                static scalar_t make_param(scalar_t log_param)
                {
                        return std::pow(10.0, log_param);
                }
                static void update_range(scalar_t& min, scalar_t& max, scalar_t opt, scalar_t delta)
                {
                        max = opt + delta;
                        min = opt - delta;
                }

                // SGD parametrization
                static scalar_t learning_rate(scalar_t gamma, scalar_t lambda, size_t iteration)
                {
                        // learning rate recommended by Bottou
                        return gamma / (1.0 + gamma * lambda * iteration);                        
                }

                // stochastic gradient descent (SGD)
                scalar_t sgd(const task_t&, const samples_t&, const loss_t&, model_t&, vector_t& x,
                             scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const;

                // learning parameter state to evaluate:
                //      <learning parameters, model copy, model parameter copy>
                struct state_t
                {
                        scalar_t        m_log_lambda;
                        scalar_t        m_log_gamma;
                        rmodel_t        m_model;
                        vector_t        m_param;
                };

        private:

                // attributes
                size_t                  m_iterations;
                size_t                  m_depth;
        };
}

#endif // STOCHASTIC_TRAINER_H
