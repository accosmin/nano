#pragma once

#include "model.h"
#include "sample.h"
#include "libnanocv/math/stats.hpp"

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

        NANOCV_PUBLIC criterion_manager_t& get_criteria();

        ///
        /// \brief accumulate sample evaluations (loss value, error and gradient),
        ///     this is the base case without regularization
        ///
        class NANOCV_PUBLIC criterion_t : public clonable_t<criterion_t>
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
                explicit criterion_t(const string_t& configuration);
                
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
                criterion_t& reset();

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss);
                void update(const vector_t& input, const vector_t& target, const loss_t& loss);

                ///
                /// \brief cumulate statistics
                ///
                criterion_t& operator+=(const criterion_t&);

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const = 0;

                ///
                /// \brief cumulated gradient
                ///
                virtual vector_t vgrad() const = 0;

                ///
                /// \brief averaged error value
                ///
                scalar_t avg_error() const;

                ///
                /// \brief variance error value
                ///
                scalar_t var_error() const;

                ///
                /// \brief total number of processed samples
                ///
                size_t count() const;

                ///
                /// \brief current parameters
                ///
                const vector_t& params() const;

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
                /// \brief reset statistics, keep parameters
                ///
                virtual void clear() = 0;

                ///
                /// \brief update statistics with the loss value/error/gradient for a sample
                ///
                virtual void accumulate(scalar_t value) = 0;
                virtual void accumulate(const vector_t& vgrad, scalar_t value) = 0;

                ///
                /// \brief update statistics with cumulated samples
                ///
                virtual void accumulate(const criterion_t& other) = 0;

                ///
                /// \brief loss term's weight
                ///
                scalar_t lweight() const { return 1.0 - lambda(); }

                ///
                /// \brief regularizer term's weight
                ///
                scalar_t rweight() const { return lambda(); }

        private:

                ///
                /// \brief update statistics with a new sample
                ///
                void accumulate(const vector_t& output, const vector_t& target, const loss_t&);

        private:

                // attributes
                rmodel_t                m_model;        ///< current model
                vector_t                m_params;       ///< current model parameters
                
                scalar_t                m_lambda;       ///< regularization weight (if any)                
                type                    m_type;         ///<

                stats_t<scalar_t>       m_estats;       ///< loss error statistics
        };
}

