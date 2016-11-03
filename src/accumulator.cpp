#include "task.h"
#include "accumulator.h"
#include "math/numeric.h"
#include "thread/loopit.h"
#include <cassert>

namespace nano
{
        struct accumulator_t::impl_t
        {
                impl_t( const model_t& model, const loss_t& loss,
                        const criterion_t& criterion, const criterion_t::type type, const scalar_t lambda) :
                        m_loss(loss)
                {
                        const auto size = thread_pool_t::instance().n_workers();
                        for (size_t i = 0; i < size; ++ i)
                        {
                                auto cache = criterion.clone();
                                cache->reset(model);
                                cache->reset(lambda);
                                cache->reset(type);
                                m_criteria.emplace_back(std::move(cache));
                        }
                }

                void cumulate() const
                {
                        for (std::size_t i = 1; i < m_criteria.size(); ++ i)
                        {
                                criterion().update(*m_criteria[i]);
                        }
                }

                criterion_t& criterion() const
                {
                        return **m_criteria.begin();
                }

                // attributes
                const loss_t&                   m_loss;
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

        void accumulator_t::set_lambda(const scalar_t ilambda) const
        {
                const auto lambda = nano::clamp(ilambda, 0.0, 1.0);
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

        void accumulator_t::set_threads(const size_t nthreads) const
        {
                thread_pool_t::instance().activate(nthreads);
        }

        void accumulator_t::update(const task_t& task, const fold_t& fold) const
        {
                update(task, fold, 0, task.n_samples(fold));
        }

        void accumulator_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
        {
                const loss_t& loss = m_impl->m_loss;

                if (thread_pool_t::instance().n_active_workers() == 1)
                {
                        m_impl->criterion().update(task, fold, begin, end, loss);
                }

                else
                {
                        loopit(end - begin, [&] (const size_t offset, const size_t th)
                        {
                                const auto index = begin + offset;
                                assert(th < m_impl->m_criteria.size());
                                assert(index < task.n_samples(fold));
                                assert(index >= begin && index < end);
                                m_impl->m_criteria[th]->update(task.input(fold, index), task.target(fold, index), loss);
                        });

                        m_impl->cumulate();
                }
        }

        scalar_t accumulator_t::value() const
        {
                return m_impl->criterion().value();
        }

        const stats_t<scalar_t>& accumulator_t::vstats() const
        {
                return m_impl->criterion().vstats();
        }

        const stats_t<scalar_t>& accumulator_t::estats() const
        {
                return m_impl->criterion().estats();
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

        model_t::timings_t accumulator_t::timings() const
        {
                model_t::timings_t ret;
                for (const auto& criterion : m_impl->m_criteria)
                {
                        const auto timings = criterion->model().timings();
                        for (const auto& timing : timings)
                        {
                                ret[timing.first](timing.second);
                        }
                }

                return ret;
        }
}

