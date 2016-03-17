#include "task.h"
#include "accumulator.h"
#include "math/clamp.hpp"
#include "thread/loopit.hpp"
#include <cassert>

namespace nano
{
        struct accumulator_t::impl_t
        {
                // constructor
                impl_t( const model_t& model, const loss_t& loss,
                        const criterion_t& criterion, const criterion_t::type type, const scalar_t lambda) :
                        m_loss(loss)
                {
                        for (size_t i = 0; i < m_pool.n_workers(); ++ i)
                        {
                                const auto cache = criterion.clone();
                                cache->reset(model);
                                cache->reset(lambda);
                                cache->reset(type);
                                m_criteria.push_back(cache);
                        }
                }

                // cumulate criterion using the results for each thread
                void cumulate() const
                {
                        for (std::size_t i = 1; i < m_criteria.size(); ++ i)
                        {
                                criterion().update(*m_criteria[i]);
                        }
                }

                // cumulated criterion
                criterion_t& criterion() const
                {
                        return **m_criteria.begin();
                }

                // attributes
                const loss_t&                   m_loss;
                nano::pool_t                    m_pool;         ///< thread pool
                std::vector<rcriterion_t>       m_criteria;     ///< cached criterion / thread
        };

        accumulator_t::accumulator_t(
                const model_t& model, const loss_t& loss,
                const criterion_t& criterion, const criterion_t::type type, const scalar_t lambda) :
                m_impl(std::make_unique<impl_t>(model, loss, criterion, type, lambda))
        {
        }

        accumulator_t::~accumulator_t() = default;

        void accumulator_t::reset() const
        {
                for (const auto& cache : m_impl->m_criteria)
                {
                        cache->reset();
                }
        }

        void accumulator_t::set_lambda(scalar_t lambda) const
        {
                lambda = nano::clamp(lambda, 0.0, 1.0);

                for (const auto& cache : m_impl->m_criteria)
                {
                        cache->reset(lambda);
                }
        }

        void accumulator_t::set_params(const vector_t& params) const
        {
                for (const auto& cache : m_impl->m_criteria)
                {
                        cache->reset(params);
                }
        }

        void accumulator_t::set_threads(size_t nthreads) const
        {
                m_impl->m_pool.activate(nthreads);
        }

        void accumulator_t::update(const task_t& task, const fold_t& fold)
        {
                return update(task, fold, 0, task.n_samples(fold));
        }

        void accumulator_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end)
        {
                const loss_t& loss = m_impl->m_loss;
                nano::loopit(end - begin, m_impl->m_pool, [&] (const size_t offset, const size_t th)
                {
                        const auto index = begin + offset;
                        const auto target = task.target(fold, index);
                        if (target.annotated())
                        {
                                m_impl->m_criteria[th]->update(task.input(fold, index), target.m_target, loss);
                        }
                });

                m_impl->cumulate();
        }

        scalar_t accumulator_t::value() const
        {
                return m_impl->criterion().value();
        }

        scalar_t accumulator_t::avg_error() const
        {
                return m_impl->criterion().avg_error();
        }

        scalar_t accumulator_t::var_error() const
        {
                return m_impl->criterion().var_error();
        }

        vector_t accumulator_t::vgrad() const
        {
                return m_impl->criterion().vgrad();
        }

        tensor_size_t accumulator_t::psize() const
        {
                return m_impl->criterion().psize();
        }

        size_t accumulator_t::count() const
        {
                return m_impl->criterion().count();
        }

        scalar_t accumulator_t::lambda() const
        {
                return m_impl->criterion().lambda();
        }

        bool accumulator_t::can_regularize() const
        {
                return m_impl->criterion().can_regularize();
        }
}

