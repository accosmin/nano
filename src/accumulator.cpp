#include "accumulator.h"
#include "thread/loopi.h"

using namespace nano;

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

void accumulator_t::random()
{
        origin().m_model->random();
        params(origin().m_model->params());
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

void accumulator_t::lambda(const scalar_t lambda)
{
        m_lambda = lambda;
        clear();
}

void accumulator_t::threads(const size_t nthreads)
{
        thread_pool_t::instance().activate(nthreads);
}

void accumulator_t::minibatch(const size_t minibatch_size)
{
        m_batch = minibatch_size;
}

void accumulator_t::update(const task_t& task, const fold_t& fold)
{
        update(task, fold, 0, task.size(fold));
}

void accumulator_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end)
{
        assert(begin <= end);
        const auto old_count = vstats().count();
        switch (thread_pool_t::instance().active_workers())
        {
        case 1:
                for (size_t chunk = std::min(end - begin, m_batch), ibegin = begin; ibegin < end; )
                {
                        const auto iend = std::min(ibegin + chunk, end);
                        update(origin(), task.get(fold, ibegin, iend));
                        ibegin = iend;
                }
                break;

        default:
                loopit(end - begin, m_batch, [&] (const size_t ibegin, const size_t iend, const size_t thread)
                {
                        assert(thread < m_tcaches.size());
                        assert(ibegin < iend && iend + begin <= end);
                        update(m_tcaches[thread], task.get(fold, begin + ibegin, begin + iend));
                });
                accumulate();
                break;
        }
        NANO_UNUSED1_RELEASE(old_count);
        assert(old_count + end == begin + vstats().count());
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
                tcache.m_vgrad += tcache.m_model->gparam(m_loss.vgrad(targets, outputs));
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

scalar_t accumulator_t::value() const
{
        assert(vstats().count() > 0);
        return vstats().avg() + (m_lambda / 2) * params().squaredNorm();
}

vector_t accumulator_t::vgrad() const
{
        assert(vstats().count() > 0);
        assert(m_type == type::vgrad);
        return origin().m_vgrad / vstats().count() + m_lambda * params();
}

tensor_size_t accumulator_t::psize() const
{
        return origin().m_model->psize();
}

const vector_t& accumulator_t::params() const
{
        return origin().m_model->params();
}

probes_t accumulator_t::probes() const
{
        return origin().m_model->probes();
}
