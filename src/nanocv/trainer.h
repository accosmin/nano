#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "model.h"
#include <cassert>

namespace ncv
{
        class trainer_t;
        class sampler_t;
        class loss_t;
        class accumulator_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;

        ///
        /// \brief track the current/optimum model state
        ///
        struct trainer_state_t
        {
                ///
                /// \brief constructor
                /// \param n_parameters
                ///
                trainer_state_t(size_t n_parameters);

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \param params
                /// \param tvalue
                /// \param terror
                /// \param vvalue
                /// \param verror
                /// \param lambda
                /// \return
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror,
                            scalar_t lambda);

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \param state
                /// \return
                ///
                bool update(const trainer_state_t& state);

                // attributes
                vector_t        m_params;       ///< current model parameters
                scalar_t        m_tvalue;       ///< train loss value (at the optimum)
                scalar_t        m_terror;       ///< train error (at the optimum)
                scalar_t        m_vvalue;       ///< optimum validation loss value
                scalar_t        m_verror;       ///< optimum validation error
                scalar_t        m_lambda;       ///< optimum regularization weight
        };
                
        ///
        /// \brief generic trainer: optimizes a model on a given task
        ///
        class trainer_t : public clonable_t<trainer_t>
        {
        public:

                trainer_t(const string_t& parameters, const string_t& description)
                        :       clonable_t<trainer_t>(parameters, description)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~trainer_t() {}

                ///
                /// \brief train the given model
                /// \param nthreads
                /// \return
                ///
                virtual bool train(const task_t&, const fold_t&, const loss_t&, size_t nthreads, model_t&) const = 0;

        protected:

                ///
                /// \brief train the given model stochastically (minibatch) or with all samples (batch)
                ///
                static bool train(
                        const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                        const loss_t&, const string_t& optimizer, size_t iterations, scalar_t epsilon,
                        const string_t& regularizer, const model_t&, trainer_state_t& state);
                static bool train(
                        const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                        const loss_t&, const string_t& optimizer, size_t iterations, scalar_t epsilon,
                        const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state);
        };
}

#endif // NANOCV_TRAINER_H
