#include "math/random.h"
#include "tensor/numeric.h"
#include "enhancer_noclass.h"

using namespace nano;

json_reader_t& enhancer_noclass_t::config(json_reader_t& reader)
{
        return reader.object("ratio", m_ratio, "noise", m_noise);
}

json_writer_t& enhancer_noclass_t::config(json_writer_t& writer) const
{
        return writer.object("ratio", m_ratio, "noise", m_noise);
}

minibatch_t enhancer_noclass_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        auto rng = make_rng();
        auto udist = make_udist<scalar_t>(-1, +1);

        minibatch_t minibatch = task.get(fold, begin, end);
        for (auto index = 0; index < minibatch.count(); ++ index)
        {
                auto&& idata = minibatch.idata(index);
                auto&& odata = minibatch.odata(index);

                if (std::fabs(udist(rng)) < m_ratio)
                {
                        // replace input with random [0, 1] values with no class label
                        nano::set_random(make_udist<scalar_t>(0, 1), rng, idata);
                        odata.constant(neg_target());
                }
                else
                {
                        // keep input and add some salt & pepper noise
                        nano::add_random(make_udist<scalar_t>(-m_noise, +m_noise), rng, idata);
                }
        }

        return minibatch;
}
