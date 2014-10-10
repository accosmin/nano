#pragma once

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
        class criterion_t : public clonable_t<criterion_t>
        {
        public:

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
                criterion_t(const string_t& configuration);
                
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
                virtual void reset() = 0;

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss, scalar_t weight = 1.0);
                void update(const vector_t& input, const vector_t& target, const loss_t& loss, scalar_t weight = 1.0);

                ///
                /// \brief cumulate statistics
                ///
                virtual criterion_t& operator+=(const criterion_t&) = 0;

                ///
                /// \brief average loss value
                ///
                virtual scalar_t value() const = 0;

                ///
                /// \brief average error value
                ///
                virtual scalar_t error() const = 0;

                ///
                /// \brief average gradient
                ///
                virtual vector_t vgrad() const = 0;
                
                ///
                /// \brief total number of processed samples
                ///
                virtual size_t count() const = 0;

                ///
                /// \brief number of dimensions/parameters
                ///
                size_t psize() const;

                ///
                /// \brief regularization weight (if any)
                ///
                scalar_t lambda() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const = 0;

        protected:

                ///
                /// \brief update statistics with a new sample
                ///
                virtual void accumulate(const vector_t& output, const vector_t& target, const loss_t&, scalar_t weight) = 0;

        protected:

                // attributes
                rmodel_t                m_model;        ///< current model
                vector_t                m_params;       ///< current model parameters
                
                scalar_t                m_lambda;       ///< regularization weight (if any)                
                type                    m_type;         ///<
        };
}

