#ifndef NANOCV_ACCUMULATOR_H
#define NANOCV_ACCUMULATOR_H

#include "task.h"
#include "model.h"
#include "common/thread_pool.h"

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
                /// \brief constructors
                ///
                accumulator_t(const model_t&, size_t nthreads, type = type::value, scalar_t lambda = 0.0);

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
                
        private:                
                
                struct config_t
                {
                        // constructor
                        config_t(type t, scalar_t lambda)
                                :       m_type(t),
                                        m_lambda(lambda)
                        {
                        }
                        
                        // attributes
                        type            m_type;
                        scalar_t        m_lambda;       ///< L2-regularization factor
                };
                
                struct data_t
                {
                        // constructor
                        data_t(size_t size = 0)
                                :       m_value(0.0),
                                        m_vgrad(size),
                                        m_error(0.0),
                                        m_count(0)
                        {
                                reset();
                        }
                        
                        // clear statistics
                        void reset()
                        {
                                m_value = 0.0;
                                m_vgrad.setZero();
                                m_error = 0.0;
                                m_count = 0;
                        }
                        
                        // cumulate statistics
                        void operator+=(const data_t& other)
                        {
                                m_value += other.m_value;
                                m_vgrad += other.m_vgrad;
                                m_error += other.m_error;
                                m_count += other.m_count;
                        }
                        
                        // attributes
                        scalar_t        m_value;        ///< cumulated loss value
                        vector_t        m_vgrad;        ///< cumulated gradient
                        scalar_t        m_error;        ///< cumulated loss error
                        size_t          m_count;        ///< #processed samples
                };
                
                struct cache_t
                {
                        // constructor
                        cache_t(size_t size = 0, type t = type::value, scalar_t lambda = 0.0)
                                :       m_config(t, lambda),
                                        m_data(size)
                        {
                        }
                        
                        // clear statistics
                        void reset(const model_t& model);
                        void reset(const vector_t& params);
                        void reset();          
                        
                        // update statistics with a new sample
                        void update(const task_t& task, const sample_t& sample, const loss_t& loss);
                        void update(const tensor_t& input, const vector_t& target, const loss_t& loss);
                        void update(const vector_t& input, const vector_t& target, const loss_t& loss);
                        void cumulate(const vector_t& output, const vector_t& target, const loss_t& loss);
                        
                        // cumulate statistics
                        void operator+=(const cache_t& other)
                        {
                                m_data += other.m_data;
                        }
                        
                        // attributes
                        rmodel_t        m_model;        ///< model copy
                        vector_t        m_params;       ///< model's parameters
                        config_t        m_config;       ///< settings
                        data_t          m_data;         ///< cumulated data                        
                };

        private:

                // attributes
                thread_pool_t           m_pool;         ///< thread pool
                std::vector<cache_t>    m_caches;       ///< cache / thread                
                cache_t                 m_cache;        ///< global (cumulated) cache
        };
}

#endif // NANOCV_ACCUMULATOR_H
