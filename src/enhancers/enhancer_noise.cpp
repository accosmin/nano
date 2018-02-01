#include "math/random.h"
#include "enhancer_noise.h"
#include "tensor/numeric.h"

using namespace nano;

json_reader_t& enhancer_noise_t::config(json_reader_t& reader)
{
        return reader.object("noise", m_noise);
}

json_writer_t& enhancer_noise_t::config(json_writer_t& writer) const
{
        return writer.object("noise", m_noise);
}

minibatch_t enhancer_noise_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        auto rng = make_rng();
        auto udist = make_udist<scalar_t>(-m_noise, +m_noise);

        minibatch_t minibatch = task.get(fold, begin, end);
        for (auto index = 0; index < minibatch.count(); ++ index)
        {
                auto&& idata = minibatch.idata(index);

                // keep input and add some salt & pepper noise
                nano::add_random(udist, rng, idata);
        }

        return minibatch;
}
