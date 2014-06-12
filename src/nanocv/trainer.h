#ifndef NANOCV_TRAINER_H
#define NANOCV_TRAINER_H

#include "task.h"

namespace ncv
{
        class trainer_t;
        class loss_t;
        class sampler_t;
        class model_t;
        struct trainer_result_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;

        ///
        /// \brief batch-train the given model
        ///
        bool batch_train(
                const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, batch_optimizer optimizer, 
                size_t cycles, size_t epochs, size_t iterations, scalar_t epsilon,
                const model_t& model, trainer_result_t& result);

        ///
        /// \brief stochastic-train the given model
        ///
        bool stochastic_train(
                const task_t&, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t&, stochastic_optimizer optimizer, size_t epochs,
                const model_t& model, trainer_result_t& result);
        
        ///
        /// \brief training state
        ///
        struct trainer_state_t
        {
                ///
                /// \brief constructor
                ///
                trainer_state_t(scalar_t tvalue = std::numeric_limits<scalar_t>::max(),
                                scalar_t terror = std::numeric_limits<scalar_t>::max(),
                                scalar_t vvalue = std::numeric_limits<scalar_t>::max(),
                                scalar_t verror = std::numeric_limits<scalar_t>::max())
                        :       m_tvalue(tvalue),
                                m_terror(terror),
                                m_vvalue(vvalue),
                                m_verror(verror)
                {
                }
                
                // attributes
                scalar_t                m_tvalue;       ///< train loss value
                scalar_t                m_terror;       ///< train error
                scalar_t                m_vvalue;       ///< validation loss value
                scalar_t                m_verror;       ///< validation error        
        };
        
        typedef std::vector
        <trainer_state_t>               trainer_states_t;
        
        ///
        /// \brief training configuration (e.g. learning rate, regularization weight)
        ///
        typedef scalars_t               trainer_config_t;
        
        ///
        /// \brief training history (configuration, optimization states)
        ///
        typedef std::map
        <
                trainer_config_t,
                trainer_states_t
        >                               trainer_history_t;
        
        ///
        /// \brief track the current/optimum model state
        ///
        struct trainer_result_t
        {
                ///
                /// \brief constructor
                ///
                trainer_result_t(size_t n_parameters, size_t epochs);

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \return true is the state was improved (aka lower validation error)
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror,
                            size_t epoch, const scalars_t& config);

                // attributes
                vector_t                m_opt_params;           ///< optimum model parameters
                trainer_state_t         m_opt_state;            ///< optimum training state
                trainer_config_t        m_opt_config;           ///< optimum configuration
                size_t                  m_opt_epoch;            ///< optimum epoch
                size_t                  m_epochs;               ///< maximum number of epochs
                trainer_history_t       m_history;              ///< optimization history
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
