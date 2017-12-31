#include "enhancer_warp.h"

using namespace nano;

json_reader_t& enhancer_warp_t::config(json_reader_t& reader)
{
        return reader.object("type", m_type, "noise", m_noise, "sigma", m_sigma, "alpha", m_alpha, "beta", m_beta);
}

json_writer_t& enhancer_warp_t::config(json_writer_t& writer) const
{
        return writer.object(
                "type", m_type, "types", join(enum_values<warp_type>()),
                "noise", m_noise, "sigma", m_sigma, "alpha", m_alpha, "beta", m_beta);
}

minibatch_t enhancer_warp_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        minibatch_t minibatch = task.get(fold, begin, end);
        warp(minibatch.idata(), m_type, m_noise, m_sigma, m_alpha, m_beta);

        return minibatch;
}
