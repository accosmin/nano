#include "accumulator.h"
#include "thread/loopit.h"
#include <cassert>

namespace nano
{
        accumulator_t::accumulator_t(const model_t& model, const loss_t& loss, const task_t& task, const sampler_t& sampler) :
                m_type(type::value), m_loss(loss), m_task(task), m_sampler(sampler)
        {
                const auto size = thread_pool_t::instance().n_workers();
                for (size_t i = 0; i < size; ++ i)
                {
                        m_tcaches.emplace_back(model);
                }
        }

        void accumulator_t::clear()
        {
                for (auto& tcache : m_tcaches)
                {
                        tcache.m_vstats.clear();
                        tcache.m_estats.clear();
                        if (m_type == type::vgrad)
                        {
                                tcache.m_vgrad.setZero();
                        }
                }
        }

        void accumulator_t::params(const vector_t& params)
        {
                for (auto& tcache : m_tcaches)
                {
                        tcache.m_model->params(params);
                }
                clear();
        }

        void accumulator_t::mode(const accumulator_t::type t)
        {
                m_type = t;
                clear();
        }

        void accumulator_t::threads(const size_t nthreads)
        {
                thread_pool_t::instance().activate(nthreads);
        }

        void accumulator_t::update(const fold_t& fold)
        {
                update(fold, 0, m_task.size(fold));
        }

        void accumulator_t::update(const fold_t& fold, const size_t begin, const size_t end)
        {
                if (thread_pool_t::instance().n_active_workers() == 1)
                {
                        for (size_t index = begin; index < end; ++ index)
                        {
                                update(origin(), fold, index);
                        }
                }

                else
                {
                        loopit(end - begin, [&] (const size_t offset, const size_t th)
                        {
                                const auto index = begin + offset;
                                assert(th < m_tcaches.size());
                                assert(index < m_task.size(fold));
                                assert(index >= begin && index < end);
                                update(m_tcaches[th], fold, index);
                        });

                        accumulate();
                }
        }

        void accumulator_t::update(tcache_t& tcache, const fold_t& fold, const size_t index)
        {
                const auto input = m_sampler.input(m_task, fold, index);
                const auto output = tcache.m_model->output(input);
                const auto target = m_sampler.target(m_task, fold, index);

                const auto value = m_loss.value(target.vector(), output.vector());
                const auto error = m_loss.error(target.vector(), output.vector());

                tcache.m_vstats(value);
                tcache.m_estats(error);
                if (m_type == type::vgrad)
                {
                        tcache.m_vgrad += tcache.m_model->gparam(m_loss.vgrad(target.vector(), output.vector()));
                }
        }

        void accumulator_t::accumulate()
        {
                auto& origin = this->origin();
                for (const auto& tcache : m_tcaches)
                {
                        if (&tcache != &origin)
                        {
                                origin.m_vstats(tcache.m_vstats);
                                origin.m_estats(tcache.m_estats);
                                if (m_type == type::vgrad)
                                {
                                        origin.m_vgrad += tcache.m_vgrad;
                                }
                        }
                }
        }

        accumulator_t::tcache_t& accumulator_t::origin()
        {
                return *m_tcaches.begin();
        }

        const accumulator_t::tcache_t& accumulator_t::origin() const
        {
                return *m_tcaches.cbegin();
        }

        const stats_t<scalar_t>& accumulator_t::vstats() const
        {
                return origin().m_vstats;
        }

        const stats_t<scalar_t>& accumulator_t::estats() const
        {
                return origin().m_estats;
        }

        vector_t accumulator_t::vgrad() const
        {
                assert(vstats().count() > 0);
                return origin().m_vgrad / vstats().count();
        }

        tensor_size_t accumulator_t::psize() const
        {
                return origin().m_model->psize();
        }

        timings_t accumulator_t::timings() const
        {
                return origin().m_model->timings();
        }
}
