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

                // SGD parametrization
                static scalar_t make_lambda(scalar_t log_lambda)
                {
                        return std::pow(10.0, log_lambda);
                }
                static scalar_t learning_rate(scalar_t gamma, scalar_t lambda, size_t iteration)
                {
                        return gamma / (1.0 + gamma * lambda * iteration);
                }

                // stochastic gradient descent (SGD)
                scalar_t sgd(const task_t&, const samples_t&, const loss_t&, model_t&, vector_t& x,
                             scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const;

                // learning parameter state to evaluate:
                //      <learning parameter value, model copy, model parameter copy, loss value>
                struct state_t
                {
                        scalar_t        m_log_lambda;
                        rmodel_t        m_model;
                        vector_t        m_param;
                        scalar_t        m_lvalue;
                };

        private:

                // attributes
                size_t                  m_iterations;
                size_t                  m_depth;
        };
}

#endif // STOCHASTIC_TRAINER_H
