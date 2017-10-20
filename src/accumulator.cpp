#include "accumulator.h"
#include "thread/loopi.h"

using namespace nano;

static const size_t max_minibatch_size = 1024;

accumulator_t::accumulator_t(const model_t& model, const loss_t& loss) :
        m_type(type::value), m_loss(loss)
{
        const auto size = thread_pool_t::instance().workers();
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

void accumulator_t::update(const task_t& task, const fold_t& fold)
{
        update(task, fold, 0, task.size(fold));
}

void accumulator_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end)
{
        loopit(end - begin, max_minibatch_size, [&] (const size_t ibegin, const size_t iend, const size_t thread)
        {
                assert(thread < m_tcaches.size());
                assert(begin <= ibegin && ibegin < iend && iend <= end);
                update(m_tcaches[thread], task.get(fold, ibegin, iend));
        });
        accumulate();
}

void accumulator_t::update(const enhancer_t& enhancer, const task_t& task, const fold_t& fold)
{
        return update(enhancer, task, fold, 0, task.size(fold));
}

void accumulator_t::update(const enhancer_t& enhancer, const task_t& task, const fold_t& fold,
        const size_t begin, const size_t end)
{
        loopit(end - begin, max_minibatch_size, [&] (const size_t ibegin, const size_t iend, const size_t thread)
        {
                assert(thread < m_tcaches.size());
                assert(begin <= ibegin && ibegin < iend && iend <= end);
                update(m_tcaches[thread], enhancer.get(task, fold, ibegin, iend));
        });
        accumulate();
}

void accumulator_t::update(tcache_t& tcache, const minibatch_t& minibatch)
{
        update(tcache, minibatch.odata(), minibatch.idata());
}

void accumulator_t::update(tcache_t& tcache, const tensor4d_t& targets, const tensor4d_t& inputs)
{
        const auto& outputs = tcache.m_model->output(inputs);

        const auto values = m_loss.value(targets, outputs);
        const auto errors = m_loss.error(targets, outputs);

        assert(outputs.size<0>() == values.size<0>());
        assert(outputs.size<0>() == errors.size<0>());

        tcache.m_vstats(values.data(), values.data() + values.size());
        tcache.m_estats(errors.data(), errors.data() + errors.size());

        if (m_type == type::vgrad)
        {
                tcache.m_vgrad += tcache.m_model->gparam(m_loss.vgrad(targets, outputs)).vector();
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

probes_t accumulator_t::probes() const
{
        return origin().m_model->probes();
}

vector_t accumulator_t::params() const
{
        return origin().m_model->params();
}
