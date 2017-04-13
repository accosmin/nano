#pragma once

#include "model.h"

namespace nano
{
        struct loss_t;
        struct task_t;
        struct fold_t;

        ///
        /// \brief stores registered prototypes
        ///
        class criterion_t;
        using criterion_manager_t = manager_t<criterion_t>;
        using rcriterion_t = criterion_manager_t::trobject;

        NANO_PUBLIC criterion_manager_t& get_criteria();

        ///
        /// \brief accumulate sample evaluations (loss value, error and gradient),
        ///     this is the base case without regularization
        ///
        class NANO_PUBLIC criterion_t : public configurable_t
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
                /// \brief enable copying
                ///
                criterion_t(const criterion_t&);
                criterion_t& operator=(const criterion_t&) = delete;

                ///
                /// \brief enable moving
                ///
                criterion_t(criterion_t&&) = default;
                criterion_t& operator=(criterion_t&&) = default;

                ///
                /// \brief create a copy of the current object
                ///
                virtual rcriterion_t clone() const = 0;

                ///
                /// \brief reset statistics (but keeps parameters)
                ///
                virtual void clear();

                ///
                /// \brief change settins (and reset statistics)
                ///
                void model(const model_t& model);
                void params(const vector_t& params);
                void lambda(const scalar_t lambda);
                void mode(const type);

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const tensor3d_t& input, const tensor3d_t& target, const loss_t& loss);

                ///
                /// \brief update statistics with a set of new samples
                ///
                void update(const task_t& task, const fold_t&, const loss_t&);
                void update(const task_t& task, const fold_t&, const size_t begin, const size_t end, const loss_t&);

                ///
                /// \brief update statistics with cumulated samples
                ///
                criterion_t& update(const criterion_t&);

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const = 0;

                ///
                /// \brief cumulated gradient
                ///
                virtual vector_t vgrad() const = 0;

                ///
                /// \brief loss function values
                ///
                const stats_t<scalar_t>& vstats() const;

                ///
                /// \brief error function values
                ///
                const stats_t<scalar_t>& estats() const;

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
                tensor_size_t psize() const;

                ///
                /// \brief regularization weight (if any)
                ///
                scalar_t lambda() const;

                ///
                /// \brief cached model
                ///
                const model_t& model() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const = 0;

        protected:

                ///
                /// \brief update statistics with the loss value/error/gradient for a sample
                ///
                virtual void accumulate(const scalar_t value) = 0;
                virtual void accumulate(const vector_t& vgrad, const scalar_t value) = 0;

                ///
                /// \brief update statistics with cumulated samples
                ///
                virtual void accumulate(const criterion_t& other) = 0;

                ///
                /// \brief loss term's weight
                ///
                scalar_t lweight() const { return 1 - lambda(); }

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
                criterion_t::type       m_type;         ///<
                stats_t<scalar_t>       m_vstats;       ///< statistics for the loss function values
                stats_t<scalar_t>       m_estats;       ///< statistics for the error function values
        };
}

