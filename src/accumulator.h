#pragma once

#include "protocol.h"
#include "criterion.h"

namespace nano
{
        ///
        /// \brief cumulate sample evaluations (loss value, error and gradient)
        ///
        class NANO_PUBLIC accumulator_t
        {
        public:

                ///
                /// \brief constructor
                ///
                accumulator_t(const model_t&, const loss_t& loss, const criterion_t&);

                ///
                /// \brief disable copying
                ///
                accumulator_t(const accumulator_t&) = delete;
                accumulator_t& operator=(const accumulator_t&) = delete;

                ///
                /// \brief enable moving
                ///
                accumulator_t(accumulator_t&&) = default;
                accumulator_t& operator=(accumulator_t&&) = default;

                ///
                /// \brief destructor
                ///
                ~accumulator_t();

                ///
                /// \brief resets accumulator (but keeps settings)
                ///
                void clear() const;

                ///
                /// \brief change settings (and resets accumulator)
                ///
                void threads(const size_t nthreads) const;
                void params(const vector_t& params) const;
                void lambda(const scalar_t lambda) const;
                void mode(const criterion_t::type) const;

                ///
                /// \brief cumulate statistics with a set of samples
                ///
                void update(const task_t&, const fold_t&) const;
                void update(const task_t&, const fold_t&, const size_t begin, const size_t end) const;

                ///
                /// \brief cumulated loss value
                ///
                scalar_t value() const;

                ///
                /// \brief cumulated gradient
                ///
                vector_t vgrad() const;

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
                /// \brief number of dimensions/parameters
                ///
                tensor_size_t psize() const;

                ///
                /// \brief regularization weight (if any)
                ///
                scalar_t lambda() const;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                bool can_regularize() const;

                ///
                /// \brief retrieve timing information (in microseconds) regarding various components
                ///     for the three basic operations (output, gradient wrt parameters, gradient wrt inputs)
                ///     by aggregating information from all cached models (if multi-threaded)
                ///
                model_t::timings_t timings() const;

        private:

                // attributes
                struct impl_t;
                std::unique_ptr<impl_t> m_impl;
        };
}
