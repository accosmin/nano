#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "model.h"
#include <cassert>

namespace ncv
{
        class trainer_t;
        class loss_t;
        class accumulator_t;
        struct trainer_state_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;

        ///
        /// \brief batch-train the given model
        ///
        opt_state_t batch_train(
                const task_t&, const samples_t& tsamples, const samples_t& vsamples, size_t nthreads,
                const loss_t&, batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon,
                const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state);

        ///
        /// \brief stochastic-train the given model
        ///
        opt_state_t stochastic_train(
                const task_t&, const samples_t& tsamples, const samples_t& vsamples, size_t nthreads,
                const loss_t&, stochastic_optimizer optimizer, size_t epochs,
                const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state);

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
                /// \return true is the state was improved (aka lower validation error)
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror,
                            scalar_t lambda, size_t fcalls, size_t gcalls);

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
                size_t          m_fcalls;       ///< loss function calls
                size_t          m_gcalls;       ///< loss gradient calls
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
        };
}

#endif // NANOCV_TRAINER_H
