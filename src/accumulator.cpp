#include "accumulator.h"
#include "thread/loopit.h"
#include <cassert>

using namespace nano;

nano::accumulator_t::accumulator_t(const model_t& model, const loss_t& loss) :
        m_type(type::value), m_loss(loss)
{
        const auto size = thread_pool_t::instance().n_workers();
        for (size_t i = 0; i < size; ++ i)
        {
                m_tcaches.emplace_back(model);
        }
}

void nano::accumulator_t::clear()
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

void nano::accumulator_t::params(const vector_t& params)
{
        for (auto& tcache : m_tcaches)
        {
                tcache.m_model->params(params);
        }
        clear();
}

void nano::accumulator_t::mode(const nano::accumulator_t::type t)
{
        m_type = t;
        clear();
}

void nano::accumulator_t::threads(const size_t nthreads)
{
        thread_pool_t::instance().activate(nthreads);
}

void nano::accumulator_t::update(const task_t& task, const fold_t& fold)
{
        update(task, fold, 0, task.size(fold));
}

void nano::accumulator_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end)
{
        switch (thread_pool_t::instance().n_active_workers())
        {
        case 1:
                for (size_t index = begin; index < end; ++ index)
                {
                        const auto sample = task.get(fold, index);
                        update(origin(), sample.m_input, sample.m_target);
                }
                break;

        default:
                loopit(end - begin, [&] (const size_t offset, const size_t th)
                {
                        const auto index = begin + offset;
                        assert(th < m_tcaches.size());
                        assert(index >= begin && index < end);
                        const auto sample = task.get(fold, index);
                        update(m_tcaches[th], sample.m_input, sample.m_target);
                });
                accumulate();
                break;
        }
}

void nano::accumulator_t::update(const iterator_t& it, const task_t& task, const fold_t& fold)
{
        return update(it, task, fold, 0, task.size(fold));
}

void nano::accumulator_t::update(const iterator_t& it, const task_t& task, const fold_t& fold,
        const size_t begin, const size_t end)
{
        switch (thread_pool_t::instance().n_active_workers())
        {
        case 1:
                for (size_t index = begin; index < end; ++ index)
                {
                        const auto sample = it.get(task, fold, index);
                        update(origin(), sample.m_input, sample.m_target);
                }
                break;

        default:
                loopit(end - begin, [&] (const size_t offset, const size_t th)
                {
                        const auto index = begin + offset;
                        assert(th < m_tcaches.size());
                        assert(index >= begin && index < end);
                        const auto sample = it.get(task, fold, index);
                        update(m_tcaches[th], sample.m_input, sample.m_target);
                });
                accumulate();
                break;
        }
}

void nano::accumulator_t::update(tcache_t& tcache, const tensor3d_t& input, const tensor3d_t& target)
{
        const auto output = tcache.m_model->output(input);

        const auto value = m_loss.value(target.vector(), output.vector());
        const auto error = m_loss.error(target.vector(), output.vector());

        tcache.m_vstats(value);
        tcache.m_estats(error);
        if (m_type == type::vgrad)
        {
                tcache.m_vgrad += tcache.m_model->gparam(m_loss.vgrad(target.vector(), output.vector()));
        }
}

void nano::accumulator_t::accumulate()
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

nano::accumulator_t::tcache_t& nano::accumulator_t::origin()
{
        return *m_tcaches.begin();
}

const nano::accumulator_t::tcache_t& nano::accumulator_t::origin() const
{
        return *m_tcaches.cbegin();
}

const stats_t<scalar_t>& nano::accumulator_t::vstats() const
{
        return origin().m_vstats;
}

const stats_t<scalar_t>& nano::accumulator_t::estats() const
{
        return origin().m_estats;
}

vector_t nano::accumulator_t::vgrad() const
{
        assert(vstats().count() > 0);
        return origin().m_vgrad / vstats().count();
}

tensor_size_t nano::accumulator_t::psize() const
{
        return origin().m_model->psize();
}

probes_t nano::accumulator_t::probes() const
{
        return origin().m_model->probes();
}
