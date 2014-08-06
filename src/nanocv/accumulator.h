#ifndef NANOCV_ACCUMULATOR_H
#define NANOCV_ACCUMULATOR_H

#include "criterion.h"

namespace ncv
{
        struct accumulator_impl_t;
        typedef std::shared_ptr<accumulator_impl_t> raccumulator_impl_t;

        ///
        /// \brief cumulate sample evaluations (loss value, error and gradient)
        ///
        class accumulator_t
        {
        public:

                ///
                /// \brief constructors
                ///
                accumulator_t(const model_t&, size_t nthreads, 
                              const string_t& criterion_name, criterion_t::type, scalar_t lambda = 0.0);

                ///
                /// \brief disable copying
                ///
                accumulator_t(const accumulator_t& other) = delete;
                accumulator_t& operator=(const accumulator_t& other) = delete;
                
                ///
                /// \brief reset statistics and settings
                ///
                void reset();
                void reset(const vector_t& params);
                void reset(scalar_t lambda);

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
                /// \brief average loss value
                ///
                scalar_t value() const;

                ///
                /// \brief average error value
                ///
                scalar_t error() const;

                ///
                /// \brief average gradient
                ///
                vector_t vgrad() const;
                
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

        private:

                // attributes
                raccumulator_impl_t     m_impl;
        };
}

#endif // NANOCV_ACCUMULATOR_H
