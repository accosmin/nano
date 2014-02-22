#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"
#include "loss.h"
#include "model.h"
#include <cassert>

namespace ncv
{
        class trainer_t;

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
                /// \return
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror);

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \param state
                /// \return
                ///
                bool update(const trainer_state_t& state);

                // attributes
                vector_t        m_params;       ///< current model parameters
                scalar_t        m_tvalue;       ///< train loss value
                scalar_t        m_terror;       ///< train error
                scalar_t        m_vvalue;       ///< validation loss value
                scalar_t        m_verror;       ///< validation error
        };

        ///
        /// \brief cumulate sample evaluations (loss value, error and gradient)
        ///
        class trainer_data_t
        {
        public:

                ///
                /// \brief processing method
                ///
                enum class type : int
                {
                        value = 0,      ///< compute loss value (faster)
                        vgrad           ///< compute loss value and gradient (slower)
                };

                ///
                /// \brief constructor
                ///
                trainer_data_t(type = type::value);

                ///
                /// \brief constructor
                ///
                trainer_data_t(const model_t&, type);

                ///
                /// \brief reset statistics and setup the given model
                ///
                void clear(const model_t& model);

                ///
                /// \brief reset statistics and keep all settings
                ///
                void clear();

                ///
                /// \brief reset statistics and change the processing method
                ///
                void clear(type t);

                ///
                /// \brief reset statistics and load the model parameters [x]
                ///
                void clear(const vector_t& x);

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss);

                ///
                /// \brief update statistics for a set of samples - single-threaded version
                ///
                void update_st(const task_t& task, const samples_t& samples, const loss_t& loss);

                ///
                /// \brief update statistics for a set of samples - multi-threaded version
                ///
                void update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads = 0);

                ///
                /// \brief update statistics with another instance
                ///
                trainer_data_t& operator+=(const trainer_data_t& other);

                ///
                /// \brief average loss value
                ///
                scalar_t value() const { return m_value / count(); }

                ///
                /// \brief average error value
                ///
                scalar_t error() const { return m_error / count(); }

                ///
                /// \brief average model gradient
                ///
                vector_t vgrad() const { return m_vgrad / count(); }

                ///
                /// \brief number of model parameters
                ///
                size_t n_parameters() const { assert(m_model); return m_model->n_parameters(); }

                ///
                /// \brief total number of processed samples
                ///
                size_t count() const { return m_count; }

        private:

                // attributes
                type                    m_type;
                scalar_t                m_value;        ///< cumulated loss value
                scalar_t                m_error;        ///< cumulated loss error
                vector_t                m_vgrad;        ///< cumulated model gradient
                size_t                  m_count;        ///< #processed samples
                rmodel_t                m_model;        ///< current model
        };
                
        ///
        /// \brief generic trainer: optimizes a model on a given task
        ///
        class trainer_t : public clonable_t<trainer_t>
        {
        public:

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
