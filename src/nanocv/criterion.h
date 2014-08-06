#ifndef NANOCV_CRITERION_H
#define NANOCV_CRITERION_H

#include "model.h"
#include "sample.h"

namespace ncv
{        
        class criterion_t;
        class loss_t;
        class task_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<criterion_t>                  criterion_manager_t;
        typedef criterion_manager_t::robject_t          rcriterion_t;

        ///
        /// \brief accumulate sample evaluations (loss value, error and gradient),
        ///     this is the base case without regularization
        ///
        class criterion_t : clonable_t<criterion_t>
        {
        public:
                
                using clonable_t<criterion_t>::robject_t;
                
                NANOCV_MAKE_CLONABLE(criterion_t)

                ///
                /// \brief processing method
                ///
                enum class type : int
                {
                        value = 0,              ///< compute loss value (faster)
                        vgrad                   ///< compute loss value and gradient (slower)
                };

                ///
                /// \brief constructor
                ///
                criterion_t(const string_t& configuration = string_t(),
                            const string_t& description = string_t());
                
                ///
                /// \brief destructor
                ///
                virtual ~criterion_t() {}
                
                ///
                /// \brief reset statistics and settings
                ///
                criterion_t& reset(const rmodel_t& rmodel);
                criterion_t& reset(const model_t& model);
                criterion_t& reset(const vector_t& params);
                criterion_t& reset(type t);
                criterion_t& reset(scalar_t lambda);
                virtual void reset();

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss);
                void update(const vector_t& input, const vector_t& target, const loss_t& loss);

                ///
                /// \brief cumulate statistics
                ///
                virtual criterion_t& operator+=(const criterion_t&);

                ///
                /// \brief average loss value
                ///
                virtual scalar_t value() const;

                ///
                /// \brief average error value
                ///
                virtual scalar_t error() const;

                ///
                /// \brief average gradient
                ///
                virtual vector_t vgrad() const;
                
                ///
                /// \brief total number of processed samples
                ///
                size_t count() const;

                ///
                /// \brief number of dimensions
                ///
                size_t dimensions() const;

                ///
                /// \brief regularization weight (if any)
                ///
                scalar_t lambda() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const;

        protected:

                ///
                /// \brief update statistics with a new sample
                ///
                virtual void cumulate(const vector_t& input, const vector_t& target, const loss_t& loss);

        protected:

                // attributes
                rmodel_t                m_model;        ///< current model
                vector_t                m_params;       ///< current model parameters
                scalar_t                m_lambda;       ///< regularization weight (if any)

                type                    m_type;         ///<
                scalar_t                m_value;        ///< cumulated loss value
                vector_t                m_vgrad;        ///< cumulated gradient
                scalar_t                m_error;        ///< cumulated loss error
                size_t                  m_count;        ///< #processed samples
        };
}

#endif // NANOCV_CRITERION_H
