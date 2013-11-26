#ifndef STOCHASTIC_TRAINER_H
#define STOCHASTIC_TRAINER_H

#include "trainer.h"
#include <mutex>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // stochastic trainer: the gradient update is computed on a random sample.
        //      the parameters are tuned in epochs of <batch> fixed size.
        //
        // parameters:
        //      batch=1024[100,10000]           - mini-batch size / epoch
        //      epoch=256[8,1024]               - number of epochs
        /////////////////////////////////////////////////////////////////////////////////////////

        class stochastic_trainer_t : public trainer_t
        {
        public:

                // constructor
                stochastic_trainer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(stochastic_trainer_t, trainer_t,
                                  "stochastic trainer, parameters: batch=1024[100,10000],epoch=256[8,1024]")

                // train the model
                virtual bool train(const task_t&, const fold_t&, const loss_t&, model_t&) const;

        private:

                // SGD parameters
                static scalar_t log_min_lambda() { return -6.0; }
                static scalar_t log_max_lambda() { return +0.0; }
                static scalar_t make_lambda(scalar_t log_lambda) { return std::pow(10.0, log_lambda); }

                static scalar_t learning_rate(scalar_t gamma, scalar_t lambda, size_t iteration)
                {
                        return gamma / (1.0 + gamma * lambda * iteration);
                }

                // stochastic gradient descent (SGD)
                void sgd(const task_t&, const samples_t&, const loss_t&, const model_t&,
                         vector_t& x, scalar_t gamma, scalar_t lambda, size_t iterations) const;

//                // A:
//                // [amin, amax, adelta, x0, #iterations]
//                // => choose the best one abest and split the [abest-adelta, abest+adelta, adelta/2, x0, 2 * #iterations]

//                // B:
//                // always iterate a fixed number (e.g. 1024 iterations)
//                // enqueue [-6, -5, -4, -3, -2, -1, 0] with delta = 1.0 and x0 starting point
//                // process each one and store for each (delta/2, loss value on random N=4096 samples, lambda)
//                // each worker should take best available lambda and process it with
//                //      lambda, lamda - delta/2, lambda + delta/2 and delta := delta/2

//                struct state_t
//                {
////                        data_t()
////                                :       m_gamma(1.0),
////                                  m_lambda()

//                        scalar_t learning_rate() const
//                        {
//                                return m_gamma / (1.0 + m_gamma * m_lambda * m_iteration);
//                        }

//                        void update(const vector_t& g)
//                        {
//                                m_x -= learning_rate() * g;
//                                m_iteration ++;
//                        }


//                        scalar_t        m_gamma;
//                        scalar_t        m_lambda, m_lambda_log_delta;
//                        size_t          m_iteration;
//                        vector_t        m_x;
//                };

//                typedef std::map
//                <
//                        scalar_t,       // loss value
//                        state_t         // stochastic setting
//                > candidates_t;

        private:

                // attributes
                size_t          m_batchsize;
                size_t          m_epochs;

//                candidates_t    m_states;
//                std::mutex      m_mutex;
        };
}

#endif // STOCHASTIC_TRAINER_H
