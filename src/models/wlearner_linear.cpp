#include "task.h"
#include "version.h"
#include "core/tpool.h"
#include "core/numeric.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "wlearner_linear.h"

using namespace nano;

void wlearner_linear_t::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients)
{
        assert(cat_dims(task.size(fold), task.odims()) == gradients.dims());

        const auto& tpool = tpool_t::instance();

        std::vector<wlearner_linear_t> learners(tpool.workers());
        scalars_t tvalues(tpool.workers(), std::numeric_limits<scalar_t>::max());

        loopit(nano::size(task.idims()), [&] (const tensor_size_t feature, const size_t t)
        {
                wlearner_linear_t learner;
                const auto value = learner.fit(task, fold, gradients, feature);
                if (value < tvalues[t])
                {
                        tvalues[t] = value;
                        std::swap(learners[t], learner);
                }
        });

        *this = learners[std::min_element(tvalues.begin(), tvalues.end()) - tvalues.begin()];
}

scalar_t wlearner_linear_t::fit(const task_t& task, const fold_t& fold, const tensor4d_t& gradients,
        const tensor_size_t feature)
{
        scalar_t x1 = 0, x2 = 0;
        tensor3d_t r1(task.odims()); r1.zero();
        tensor3d_t r2(task.odims()); r2.zero();
        tensor3d_t rx(task.odims()); rx.zero();

        for (size_t i = 0, size = task.size(fold); i < size; ++ i)
        {
                const auto input = task.input(fold, i);
                const auto value = input(feature);

                x1 += value;
                x2 += value * value;
                r1.array() -= gradients.array(i);
                rx.array() -= value * gradients.array(i);
                r2.array() += gradients.array(i) * gradients.array(i);
        }

        m_a.resize(task.odims());
        m_b.resize(task.odims());

        const auto count = static_cast<scalar_t>(task.size(fold));
        m_a.array() = (rx.array() * count - x1 * r1.array()) / (x2 * count - nano::square(x1));
        m_b.array() = (r1.array() * x2 - x1 * rx.array()) / (x2 * count - nano::square(x1));

        // todo: implement subsampling

        return  (m_a.array() * x2 + count * m_b.array().square() + r2.array()
                +2 * m_a.array() * m_b.array() * x1
                -2 * m_b.array() * r1.array()
                -2 * m_a.array() * rx.array()).sum();
}

bool wlearner_linear_t::save(obstream_t& stream) const
{
        const auto vmajor = static_cast<uint8_t>(major_version);
        const auto vminor = static_cast<uint8_t>(minor_version);

        return  stream.write(vmajor) &&
                stream.write(vminor) &&
                stream.write(m_feature) &&
                stream.write_tensor(m_a) &&
                stream.write_tensor(m_b);
}

bool wlearner_linear_t::load(ibstream_t& stream)
{
        uint8_t vmajor = 0x00;
        uint8_t vminor = 0x00;

        return  stream.read(vmajor) &&
                stream.read(vminor) &&
                stream.read(m_feature) &&
                stream.read_tensor(m_a) &&
                stream.read_tensor(m_b);
}
