#include "task_peak2d.h"
#include "math/numeric.h"

using namespace nano;

peak2d_task_t::peak2d_task_t() :
        mem_tensor_task_t(make_dims(1, 32, 32), make_dims(2, 1, 1), 1)
{
}

json_reader_t& peak2d_task_t::config(json_reader_t& reader)
{
        reader.object("irows", m_irows, "icols", m_icols, "noise", m_noise, "count", m_count, "type", m_type);
        reconfig(make_dims(1, m_irows, m_icols), make_dims(2, 1, 1), 1);
        return reader;
}

json_writer_t& peak2d_task_t::config(json_writer_t& writer) const
{
        return writer.object("irows", m_irows, "icols", m_icols, "noise", m_noise, "count", m_count,
                "type", m_type, "types", join(enum_values<peak2d_task_type>() ));
}

bool peak2d_task_t::populate()
{
        auto rng = make_rng();
        auto udist_noise = make_udist<scalar_t>(-m_noise, +m_noise);
        auto udist_peak = make_udist<scalar_t>(-1, +1);

        tensor3d_t input(1, m_irows, m_icols);
        tensor3d_t target(2, 1, 1);

        // generate samples
        reserve_chunks(m_count);
        for (size_t i = 0; i < m_count; ++ i)
        {
                const auto peakx = udist_peak(rng);
                const auto peaky = udist_peak(rng);

                const auto halfx = static_cast<scalar_t>(m_icols) / 2;
                const auto halfy = static_cast<scalar_t>(m_irows) / 2;

                for (tensor_size_t y = 0; y < m_irows; ++ y)
                {
                        for (tensor_size_t x = 0; x < m_icols; ++ x)
                        {
                                const auto dx = static_cast<scalar_t>(x) / halfx - 1;   // [-1, +1]
                                const auto dy = static_cast<scalar_t>(y) / halfy - 1;   // [-1, +1]

                                input(0, y, x) = square(dx + peakx) + square(dy + peaky) + udist_noise(rng);
                        }
                }

                switch (m_type)
                {
                case peak2d_task_type::regression:
                        target(0) = peakx;
                        target(1) = peaky;
                        break;

                case peak2d_task_type::classification:
                        target(0) = (peakx < 0) ? neg_target() : pos_target();
                        target(1) = (peaky < 0) ? neg_target() : pos_target();
                        break;

                default:
                        assert(false);
                }

                const auto hash = i;
                const auto label = string_t();

                add_chunk(input, hash);
                add_sample(make_fold(0), i, target, label);
        }

        return true;
}
