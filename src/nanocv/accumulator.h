#ifndef NANOCV_ACCUMULATOR_H
#define NANOCV_ACCUMULATOR_H

#include "task.h"
#include "model.h"

namespace ncv
{
        class loss_t;

        ///
        /// \brief cumulate sample evaluations (loss value, error and gradient)
        ///
        class accumulator_t
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
                /// \brief regularization method
                ///
                enum class regularizer : int
                {
                        none = 0,               ///< no regularization (sum the samples)
                        l2norm,                 ///< L2-norm regularization on the model's parameters (uses lambda)
                        variational             ///< penalize large variations between samples' performance (uses lambda)
                };

                ///
                /// \brief constructors
                ///
                accumulator_t(const model_t&,
                              type = type::value,
                              regularizer = regularizer::none,
                              scalar_t lambda = 0.0);

                accumulator_t(const rmodel_t& = rmodel_t(),
                              type = type::value,
                              regularizer = regularizer::none,
                              scalar_t lambda = 0.0);

                ///
                /// \brief copy constructor
                ///
                accumulator_t(const accumulator_t& other);

                ///
                /// \brief assignment operator
                ///
                accumulator_t& operator=(const accumulator_t& other);

                ///
                /// \brief reset statistics and settings
                ///
                void reset();
                void reset(const model_t& model);
                void reset(const vector_t& params);
                void reset(type, regularizer, scalar_t lambda);

                ///
                /// \brief update statistics with a new sample
                ///
                void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                void update(const tensor_t& input, const vector_t& target, const loss_t& loss);
                void update(const vector_t& input, const vector_t& target, const loss_t& loss);

                ///
                /// \brief update statistics for a set of samples - single-threaded version
                ///
                void update(const task_t& task, const samples_t& samples, const loss_t& loss);
                void update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss);
                void update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss);

                ///
                /// \brief update statistics for a set of samples - multi-threaded version
                ///
                void update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads = 0);
                void update_mt(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads = 0);
                void update_mt(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads = 0);

                ///
                /// \brief update statistics with another instance
                ///
                accumulator_t& operator+=(const accumulator_t& other);

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
                /// \brief number of dimensions
                ///
                size_t dimensions() const;

                ///
                /// \brief total number of processed samples
                ///
                size_t count() const;

        private:

                ///
                /// \brief accumulate the output
                ///
                void cumulate(const vector_t& output, const vector_t& target, const loss_t& loss);

        private:

                struct settings_t
                {
                        // constructor
                        settings_t(type t, regularizer r, scalar_t lambda)
                                :       m_type(t),
                                        m_regularizer(r),
                                        m_lambda(lambda)
                        {
                        }

                        // attributes
                        type            m_type;
                        regularizer     m_regularizer;
                        scalar_t        m_lambda;       ///< regularization factor (if any)
                };

                struct data_t
                {
                        // constructor
                        data_t(size_t size = 0)
                                :       m_value1(0.0),
                                        m_value2(0.0),
                                        m_grad1(size),
                                        m_grad2(size),
                                        m_error(0.0),
                                        m_count(0)
                        {
                                reset();
                        }

                        // clear statistics
                        void reset()
                        {
                                m_value1 = 0.0;
                                m_value2 = 0.0;
                                m_grad1.setZero();
                                m_grad2.setZero();
                                m_error = 0.0;
                                m_count = 0;
                        }

                        // attributes
                        scalar_t        m_value1;       ///< cumulated loss value
                        scalar_t        m_value2;       ///< cumulated squared loss value
                        vector_t        m_grad1;        ///< cumulated gradient
                        vector_t        m_grad2;        ///< cumulated loss value * gradient
                        scalar_t        m_error;        ///< cumulated loss error
                        size_t          m_count;        ///< #processed samples
                        vector_t        m_params;       ///< model's parameters
                };

                // attributes
                settings_t              m_settings;
                rmodel_t                m_model;        ///< current model
                data_t                  m_data;
        };
}

#endif // NANOCV_ACCUMULATOR_H
