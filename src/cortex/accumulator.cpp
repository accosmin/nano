#include "accumulator.h"
#include "thread/loopit.hpp"
#include <cassert>

namespace cortex
{
        struct accumulator_t::impl_t
        {
                // constructor
                impl_t(const model_t& model, const criterion_t& criterion, criterion_t::type type, scalar_t lambda)
                        :       m_cache(criterion.clone())
                {
                        m_cache->reset(model);
                        m_cache->reset(lambda);
                        m_cache->reset(type);

                        if (m_pool.n_workers() > 1)
                        {
                                for (size_t i = 0; i < m_pool.n_workers(); ++ i)
                                {
                                        const rcriterion_t cache = criterion.clone();
                                        cache->reset(model);
                                        cache->reset(lambda);
                                        cache->reset(type);
                                        m_caches.push_back(cache);
                                }
                        }
                }
                
                // attributes
                thread::pool_t                  m_pool;         ///< thread pool
                rcriterion_t                    m_cache;        ///< global (cumulated) criterion
                std::vector<rcriterion_t>       m_caches;       ///< cached criterion / thread
        };        

        accumulator_t::accumulator_t(
                const model_t& model, const criterion_t& criterion, criterion_t::type type, scalar_t lambda) :
                m_impl(std::make_unique<impl_t>(model, criterion, type, lambda))
        {
        }

        accumulator_t::~accumulator_t() = default;

        void accumulator_t::reset()
        {
                m_impl->m_cache->reset();
                for (const auto& cache : m_impl->m_caches)
                {
                        cache->reset();
                }
        }

        void accumulator_t::set_lambda(scalar_t lambda)
        {
                lambda = math::clamp(lambda, 0.0, 1.0);

                m_impl->m_cache->reset(lambda);
                for (const auto& cache : m_impl->m_caches)
                {
                        cache->reset(lambda);
                }
        }

        void accumulator_t::set_params(const vector_t& params)
        {
                m_impl->m_cache->reset(params);
                for (const auto& cache : m_impl->m_caches)
                {
                        cache->reset(params);
                }
        }

        void accumulator_t::set_threads(size_t nthreads)
        {
                m_impl->m_pool.activate(nthreads);
        }

        void accumulator_t::update(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                if (m_impl->m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < samples.size(); ++ i)
                        {
                                m_impl->m_cache->update(task, samples[i], loss);
                        }
                }

                else
                {
                        thread::loopit(samples.size(), m_impl->m_pool, [&] (size_t i, size_t th)
                        {
                                m_impl->m_caches[th]->update(task, samples[i], loss);
                        });

                        for (const auto& cache : m_impl->m_caches)
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

        tensor_size_t accumulator_t::psize() const
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

        bool accumulator_t::can_regularize() const
        {
                return m_impl->m_cache->can_regularize();
        }
}
	
