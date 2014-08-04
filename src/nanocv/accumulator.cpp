#include "accumulator.h"
#include "loss.h"
#include "common/thread_loop.hpp"
#include <cassert>

namespace ncv
{        
        struct config_t
        {
                // constructor
                config_t(accumulator_t::type t, scalar_t lambda)
                        :       m_type(t),
                                m_lambda(lambda)
                {
                }
                
                // attributes
                accumulator_t::type     m_type;
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
                cache_t(size_t size = 0, accumulator_t::type t = accumulator_t::type::value, scalar_t lambda = 0.0)
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
        
        struct accumulator_impl_t
        {
                // constructor
                accumulator_impl_t(const model_t& model, size_t nthreads, accumulator_t::type t, scalar_t lambda)
                        :       m_pool(nthreads),                        
                                m_caches(m_pool.n_workers(), { model.psize(), t, lambda }),
                                m_cache(model.psize(), t, lambda)
                {
                }
                
                // attributes
                thread_pool_t           m_pool;         ///< thread pool
                std::vector<cache_t>    m_caches;       ///< cache / thread                
                cache_t                 m_cache;        ///< global (cumulated) cache
        };
        
        void cache_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                // loss value
                m_data.m_value += loss.value(target, output);
                m_data.m_error += loss.error(target, output);
                m_data.m_count ++;
                
                // loss gradient
                switch (m_config.m_type)
                {
                        case accumulator_t::type::value:
                                break;

                        case accumulator_t::type::vgrad:
                                m_data.m_vgrad += m_model->pgrad(loss.vgrad(target, output));
                                break;
                }
        }
        
        void cache_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(sample.m_index < task.n_images());
                
                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;                
                const vector_t& output = m_model->output(image, sample.m_region).vector();
                
                cumulate(output, target, loss);
        }
        
        void cache_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->output(input).vector();
                
                cumulate(output, target, loss);
        }
        
        void cache_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->output(input).vector();
                
                cumulate(output, target, loss);
        }
        
        void cache_t::reset(const model_t& model)
        {
                m_model = model.clone();
                m_params = model.params();
                m_data.reset();
        }
        
        void cache_t::reset(const vector_t& params)
        {
                m_model->load_params(params);
                m_params = params;
                m_data.reset();
        }
        
        void cache_t::reset()
        {
                m_data.reset();
        } 
        
        accumulator_t::accumulator_t(const model_t& model, size_t nthreads, type t, scalar_t lambda)
                :       m_impl(new accumulator_impl_t(model, nthreads, t, lambda))
        {
                m_impl->m_cache.reset(model);                
                for (cache_t& cache : m_impl->m_caches)
                {
                        cache.reset(model);
                }
        }

        void accumulator_t::reset(const vector_t& params)
        {
                m_impl->m_cache.reset(params);
                for (cache_t& cache : m_impl->m_caches)
                {
                        cache.reset(params);
                }
        }

        void accumulator_t::reset()
        {
                m_impl->m_cache.reset();
                for (cache_t& cache : m_impl->m_caches)
                {
                        cache.reset();
                }
        }       

        void accumulator_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                m_impl->m_cache.update(task, sample, loss);
        }

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                m_impl->m_cache.update(input, target, loss);
        }

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                m_impl->m_cache.update(input, target, loss);
        }

        void accumulator_t::update(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                if (m_impl->m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < samples.size(); i ++)
                        {
                                update(task, samples[i], loss);
                        }
                }

                else
                {
                        thread_loopit(samples.size(), m_impl->m_pool, [&] (size_t i, size_t th)
                        {
                                m_impl->m_caches[th].update(task, samples[i], loss);
                        });
                        
                        for (const cache_t& cache : m_impl->m_caches)
                        {
                                m_impl->m_cache += cache;
                        }
                }
        }

        void accumulator_t::update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                if (m_impl->m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < inputs.size(); i ++)
                        {
                                update(inputs[i], targets[i], loss);
                        }
                }

                else
                {
                        thread_loopit(inputs.size(), m_impl->m_pool, [&] (size_t i, size_t th)
                        {
                                m_impl->m_caches[th].update(inputs[i], targets[i], loss);
                        });
                        
                        for (const cache_t& cache : m_impl->m_caches)
                        {
                                m_impl->m_cache += cache;
                        }
                }
        }

        void accumulator_t::update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                if (m_impl->m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < inputs.size(); i ++)
                        {
                                update(inputs[i], targets[i], loss);
                        }
                }

                else
                {
                        thread_loopit(inputs.size(), m_impl->m_pool, [&] (size_t i, size_t th)
                        {
                                m_impl->m_caches[th].update(inputs[i], targets[i], loss);
                        });
                        
                        for (const cache_t& cache : m_impl->m_caches)
                        {
                                m_impl->m_cache += cache;
                        }
                }
        }
        
        scalar_t accumulator_t::value() const
        {
                assert(count() > 0);

                return  m_impl->m_cache.m_data.m_value / count() +
                        0.5 * m_impl->m_cache.m_config.m_lambda / dimensions() * m_impl->m_cache.m_params.squaredNorm();
        }

        scalar_t accumulator_t::error() const
        {
                assert(count() > 0);

                return m_impl->m_cache.m_data.m_error / count();
        }

        vector_t accumulator_t::vgrad() const
        {
                assert(count() > 0);

                return  m_impl->m_cache.m_data.m_vgrad / count() +
                        m_impl->m_cache.m_config.m_lambda / dimensions() * m_impl->m_cache.m_params;
        }

        size_t accumulator_t::dimensions() const
        {
                return static_cast<size_t>(m_impl->m_cache.m_params.size());
        }

        size_t accumulator_t::count() const
        {
                return m_impl->m_cache.m_data.m_count;
        }

        scalar_t accumulator_t::lambda() const
        {
                return m_impl->m_cache.m_config.m_lambda;
        }
}
	
