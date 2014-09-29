#pragma once

#include "task.h"

namespace ncv
{
        class trainer_t;
        class loss_t;
        class sampler_t;
        class model_t;
        class accumulator_t;
        struct trainer_result_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<trainer_t>                    trainer_manager_t;
        typedef trainer_manager_t::robject_t            rtrainer_t;
        
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

        ///
        /// \brief compare two training states
        ///
        inline bool operator<(const trainer_state_t& one, const trainer_state_t& another)
        {
                return one.m_verror < another.m_verror;
        }
        
        typedef std::vector
        <trainer_state_t>               trainer_states_t;

        ///
        /// \brief save optimization states to text file
        ///
        bool save(const string_t& path, const trainer_states_t& states);
        
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
                trainer_result_t();

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \return true is the state was improved (aka lower validation error)
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror,
                            size_t epoch, const scalars_t& config);
                bool update(const trainer_result_t& other);

                ///
                /// \brief check if valid result
                ///
                bool valid() const
                {
                        return !m_history.empty() && m_opt_params.size() > 0;
                }
                
                ///
                /// \brief training history for the optimum configuration
                ///
                trainer_states_t optimum_states() const;

                // attributes
                vector_t                m_opt_params;           ///< optimum model parameters
                trainer_state_t         m_opt_state;            ///< optimum training state
                trainer_config_t        m_opt_config;           ///< optimum configuration
                size_t                  m_opt_epoch;            ///< optimum epoch
                trainer_history_t       m_history;              ///< optimization history
        };

        ///
        /// \brief compare two trainer results
        ///
        inline bool operator<(const trainer_result_t& one, const trainer_result_t& other)
        {
                return one.m_opt_state < other.m_opt_state;
        }
        
        ///
        /// \brief stores all required buffers to train a model
        ///
        struct trainer_data_t
        {        
                ///
                /// \brief constructor
                ///
                trainer_data_t(const task_t& task,
                               const sampler_t& tsampler,
                               const sampler_t& vsampler,
                               const loss_t& loss,
                               const vector_t& x0,
                               accumulator_t& lacc,
                               accumulator_t& gacc);
                
                // attributes
                const task_t&           m_task;                 ///< 
                const sampler_t&        m_tsampler;             ///< training samples
                const sampler_t&        m_vsampler;             ///< validation samples
                const loss_t&           m_loss;                 ///< 
                const vector_t&         m_x0;                   ///< starting parameters
                accumulator_t&          m_lacc;                 ///< criterion
                accumulator_t&          m_gacc;                 ///< criterion's gradient
        };
                
        ///
        /// \brief generic trainer: optimizes a model on a given task
        ///
        class trainer_t : public clonable_t<trainer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
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
                ///
                virtual trainer_result_t train(
                        const task_t&, const fold_t&, const loss_t&, size_t nthreads, const string_t& criterion, 
                        model_t&) const = 0;
        };
}

