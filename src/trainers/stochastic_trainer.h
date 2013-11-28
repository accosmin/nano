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

                // SGD parametrization
                static scalar_t log_min_lambda() { return -6.0; }
                static scalar_t log_max_lambda() { return +0.0; }
                static scalar_t make_lambda(scalar_t log_lambda) { return std::pow(10.0, log_lambda); }

                static scalar_t learning_rate(scalar_t gamma, scalar_t lambda, size_t iteration)
                {
                        return gamma / (1.0 + gamma * lambda * iteration);
                }

                // store a SGD parameter configuration
                //      useful for tuning the learning parameters
                struct sgd_state_t;
                typedef std::vector<sgd_state_t> sgd_states_t;

                struct sgd_state_t
                {
                        // constructor
                        sgd_state_t(size_t depth = 0,
                                    scalar_t gamma = 1.0,
                                    scalar_t log_lambda = 0.0,
                                    scalar_t log_dlambda = 1.0,
                                    const vector_t& x = vector_t())
                                :       m_depth(depth),
                                        m_gamma(gamma),
                                        m_log_lambda(log_lambda),
                                        m_log_dlambda(log_dlambda),
                                        m_x(x),
                                        m_fx(std::numeric_limits<scalar_t>::max())
                        {
                        }

                        // configuration to further explore
                        sgd_states_t explorables() const
                        {
                                const scalar_t delta = m_log_dlambda / 3.0;
                                return
                                {
                                        sgd_state_t(m_depth + 1, m_gamma, m_log_lambda - delta, delta, m_x),
                                        sgd_state_t(m_depth + 1, m_gamma, m_log_lambda,         delta, m_x),
                                        sgd_state_t(m_depth + 1, m_gamma, m_log_lambda + delta, delta, m_x)

                                };
                        }

                        // check if the state was explored
                        bool processed() const
                        {
                                return m_fx < 1e+40;
                        }

                        // attributes
                        size_t          m_depth;                // exploration (accuracy) depth
                        scalar_t        m_gamma;                // learning parameter
                        scalar_t        m_log_lambda;           // learning parameter
                        scalar_t        m_log_dlambda;          // current variation of the learning parameter
                        vector_t        m_x;                    // model parameter
                        scalar_t        m_fx;                   // loss value
                };

                struct sgd_world_t
                {
                        // constructor
                        sgd_world_t(size_t max_depth = 8,
                                    const vector_t& x0 = vector_t())
                                :       m_max_depth(max_depth),
                                        m_states(
                                        {
                                                {0, 1.0, -6.0, 1.0, x0},
                                                {0, 1.0, -5.0, 1.0, x0},
                                                {0, 1.0, -4.0, 1.0, x0},
                                                {0, 1.0, -3.0, 1.0, x0},
                                                {0, 1.0, -2.0, 1.0, x0},
                                                {0, 1.0, -1.0, 1.0, x0},
                                                {0, 1.0, +0.0, 1.0, x0}
                                        })
                        {
                        }

                        // enqueued new explored state
                        void enqueue(const sgd_state_t& state)
                        {
                                const std::lock_guard<std::mutex> lock(m_mutex);

                                if (state.m_depth < m_max_depth)
                                {
                                        const sgd_states_t states = state.explorables();
                                        m_states.insert(m_states.end(), states.begin(), states.end());
                                }
                                else
                                {
                                        m_states.push_back(state);
                                }
                        }

                        // check if any state can be explored
                        bool enquire(sgd_state_t& state)
                        {
                                // TODO:
                                //      - check if maximum depth was reached => STOP
                                //      - get the unexplored state with the maximum depth
                        }

                        // attributes
                        std::mutex      m_mutex;
                        size_t          m_max_depth;            // maximum exploration depth
                        sgd_states_t    m_states;               // current (explored, to explore) states
                };

                // stochastic gradient descent (SGD)
                void sgd(const task_t&, const samples_t&, const loss_t&, const model_t&,
                         vector_t& x, scalar_t gamma, scalar_t lambda, size_t iterations) const;

//                * class to store processed and enqueued states: [age = 0-8][{lambda, lambda delta, x, f(x)}] + current age + synchronization access
//                * threading algorithm (RUN): if no job available, then go to the most recent age where there is job that was not expanded and run it; otherwise, take a job from the most recent age
//                * threading algorithm (DONE): if maximum age was reached, then exit; otherwise, expand the current state (lambda - dl, l, l + dl) with dl = dl / 3


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
