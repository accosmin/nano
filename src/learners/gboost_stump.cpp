#include "gboost_stump.h"

using namespace nano;

void gboost_stump_t::to_json(json_t&) const
{
}

void gboost_stump_t::from_json(const json_t&)
{
}

trainer_result_t gboost_stump_t::train(const task_t&, const size_t, const loss_t&)
{
        trainer_result_t result;
        return result;
}

tensor4d_t gboost_stump_t::output(const tensor4d_t&) const
{
        tensor4d_t output;
        return output;
}

bool gboost_stump_t::save(obstream_t&) const
{
        return false;
}

bool gboost_stump_t::load(ibstream_t&)
{
        return false;
}

probes_t gboost_stump_t::probes() const
{
        probes_t probes;
        return probes;
}
