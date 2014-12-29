#include "accumulator.h"
#include "criterion.h"
#include "util/thread_loop.hpp"
#include <cassert>

namespace ncv
{
        struct accumulator_impl_t
        {
                // constructor
                accumulator_impl_t(const model_t& model, size_t nthreads, const string_t& criterion_name,
                                   criterion_t::type type, scalar_t lambda)
                        :       m_pool(nthreads),
                                m_cache(criterion_manager_t::instance().get(criterion_name))
                {
                        m_cache->reset(model);
                        m_cache->reset(lambda);
                        m_cache->reset(type);

                        if (m_pool.n_workers() > 1)
                        {
                                for (size_t i = 0; i < m_pool.n_workers(); i ++)
                                {
                                        const rcriterion_t cache = criterion_manager_t::instance().get(criterion_name);
                                        cache->reset(model);
                                        cache->reset(lambda);
                                        cache->reset(type);
                                        m_caches.push_back(cache);
                                }
                        }
                }
                
                // attributes
                thread_pool_t                   m_pool;         ///< thread pool
                rcriterion_t                    m_cache;        ///< global (cumulated) criterion
                std::vector<rcriterion_t>       m_caches;       ///< cached criterion / thread
        };        

        accumulator_t::accumulator_t(const model_t& model, size_t nthreads,
                                     const string_t& criterion_name, criterion_t::type type, scalar_t lambda)
                :       m_impl(new accumulator_impl_t(model, nthreads, criterion_name, type, lambda))
        {
        }

        void accumulator_t::reset(const vector_t& params)
        {
                m_impl->m_cache->reset(params);
                for (const rcriterion_t& cache : m_impl->m_caches)
                {
                        cache->reset(params);
                }
        }

        void accumulator_t::reset()
        {
                m_impl->m_cache->reset();
                for (const rcriterion_t& cache : m_impl->m_caches)
                {
                        cache->reset();
                }
        }       

        void accumulator_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                m_impl->m_cache->update(task, sample, loss);
        }

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss, scalar_t weight)
        {
                m_impl->m_cache->update(input, target, loss, weight);
        }

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss, scalar_t weight)
        {
                m_impl->m_cache->update(input, target, loss, weight);
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
                                m_impl->m_caches[th]->update(task, samples[i], loss);
                        });
                        
                        for (const rcriterion_t& cache : m_impl->m_caches)
                        {
                                (*m_impl->m_cache) += (*cache);
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
                                m_impl->m_caches[th]->update(inputs[i], targets[i], loss);
                        });
                        
                        for (const rcriterion_t& cache : m_impl->m_caches)
                        {
                                (*m_impl->m_cache) += (*cache);
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
                                m_impl->m_caches[th]->update(inputs[i], targets[i], loss);
                        });
                        
                        for (const rcriterion_t& cache : m_impl->m_caches)
                        {
                                (*m_impl->m_cache) += (*cache);
                        }
                }
        }
        
        scalar_t accumulator_t::value() const
        {
                return m_impl->m_cache->value();
        }

        scalar_t accumulator_t::avg_error() const
        {
                return m_impl->m_cache->avg_error();
        }

        scalar_t accumulator_t::var_error() const
        {
                return m_impl->m_cache->var_error();
        }

        vector_t accumulator_t::vgrad() const
        {
                return m_impl->m_cache->vgrad();
        }

        size_t accumulator_t::psize() const
        {
                return m_impl->m_cache->psize();
        }

        size_t accumulator_t::count() const
        {
                return m_impl->m_cache->count();
        }

        scalar_t accumulator_t::lambda() const
        {
                return m_impl->m_cache->lambda();
        }

        bool accumulator_t::can_regularize(const string_t& criterion)
        {
                return criterion_manager_t::instance().get(criterion)->can_regularize();
        }
}
	
