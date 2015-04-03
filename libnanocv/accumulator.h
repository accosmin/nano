#pragma once

#include "criterion.h"
#include "util/arch.h"
#include "util/stats.hpp"

namespace ncv
{
        ///
        /// \brief cumulate sample evaluations (loss value, error and gradient)
        ///
        class NANOCV_DLL_PUBLIC accumulator_t : private noncopyable_t
        {
        public:

                ///
                /// \brief constructors
                ///
                accumulator_t(const model_t&, size_t nthreads, 
                              const string_t& criterion_name, criterion_t::type, scalar_t lambda = 0.0);

                ///
                /// \brief reset statistics (keeps parameters)
                ///
                void reset();

                ///
                /// \brief change parameters (and resets statistics)
                ///
                void set_params(const vector_t& params);

                ///
                /// \brief change the regularization weight (keeps parameters)
                ///
                scalar_t set_lambda(scalar_t lambda);

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss);
                void update(const vector_t& input, const vector_t& target, const loss_t& loss);

                ///
                /// \brief update statistics for a set of samples
                ///
                void update(const task_t& task, const samples_t& samples, const loss_t& loss);
                void update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss);
                void update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss);

                ///
                /// \brief cumulated loss value
                ///
                scalar_t value() const;

                ///
                /// \brief cumulated gradient
                ///
                vector_t vgrad() const;

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
                static bool can_regularize(const string_t& criterion);

        private:

                // attributes
                struct impl_t;
                std::shared_ptr<impl_t> m_impl;
        };
}
