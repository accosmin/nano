#include "core/tpool.h"
#include "gboost_stump.h"
#include "core/ibstream.h"
#include "core/obstream.h"

using namespace nano;

void gboost_stump_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "rounds", m_rounds,
                "stump", m_stype, "stumps", join(enum_values<stump>()),
                "regularization", m_rtype, "regularizations", join(enum_values<regularization>()));
}

void gboost_stump_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "stump", m_stype,
                "regularization", m_rtype);
}

trainer_result_t gboost_stump_t::train(const task_t& task, const size_t, const loss_t&)
{
        m_idims = task.idims();
        m_odims = task.odims();

        m_stumps.clear();

        for (auto round = 0; round < m_rounds; ++ round)
        {

        }


        trainer_result_t result;
        return result;
}

tensor4d_t gboost_stump_t::output(const tensor4d_t& input) const
{
        const auto count = input.size<0>();
        assert(input.dims() == cat_dims(count, m_idims));

        tensor4d_t output(cat_dims(count, m_odims));
        output.zero();

        // todo: use the thread pool to speed-up computation
        for (auto i = 0; i < count; ++ i)
        {
                const auto idata = input.tensor(i);
                auto odata = output.tensor(i);

                for (const auto& stump : m_stumps)
                {
                        const auto oindex = idata(stump.m_feature) < stump.m_threshold ? 0 : 1;
                        odata.array() += stump.m_outputs.tensor(oindex).array();
                }
        }

        return output;
}

bool gboost_stump_t::save(obstream_t& stream) const
{
        if (    !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_stype) ||
                !stream.write(m_rtype) ||
                !stream.write(m_stumps.size()))
        {
                return false;
        }

        for (const auto& stump : m_stumps)
        {
                assert(stump.m_feature >= 0 && stump.m_feature < nano::size(m_idims));
                assert(stump.m_outputs.dims() == cat_dims(2, m_odims));

                if (    !stream.write(stump.m_feature) ||
                        !stream.write(stump.m_threshold) ||
                        !stream.write_tensor(stump.m_outputs))
                {
                       return false;
                }
        }

        return true;
}

bool gboost_stump_t::load(ibstream_t& stream)
{
        size_t n_stumps = 0;
        if (    !stream.read(m_idims) ||
                !stream.read(m_odims) ||
                !stream.read(m_rounds) ||
                !stream.read(m_stype) ||
                !stream.read(m_rtype) ||
                !stream.read(n_stumps))
        {
                return false;
        }

        m_stumps.resize(n_stumps);
        for (auto& stump : m_stumps)
        {
                if (    !stream.read(stump.m_feature) ||
                        !stream.read(stump.m_threshold) ||
                        !stream.read_tensor(stump.m_outputs) ||
                        stump.m_feature < 0 ||
                        stump.m_feature >= nano::size(m_idims) ||
                        stump.m_outputs.dims() != cat_dims(2, m_odims))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#stumps, feature or coefficient statistics, idims...)

        return true;
}

probes_t gboost_stump_t::probes() const
{
        // todo: add probes here to measure the training and the evaluation time
        probes_t probes;
        return probes;
}
