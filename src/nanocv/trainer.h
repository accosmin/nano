#pragma once

#include "task.h"
#include "trainer_state.h"
#include "trainer_result.h"

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
                trainer_t(const string_t& configuration)
                        :       clonable_t<trainer_t>(configuration)
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

