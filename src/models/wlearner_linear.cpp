#include "task.h"
#include "core/tpool.h"
#include "core/numeric.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "wlearner_linear.h"

using namespace nano;

void wlearner_linear_t::fit(const task_t& task, const fold_t& fold, const tensor4d_t& residuals,
        const wlearner_type type)
{
        assert(cat_dims(task.size(fold), task.odims()) == residuals.dims());

        const auto& tpool = tpool_t::instance();

        std::vector<wlearner_linear_t> learners(tpool.workers());
        scalars_t tvalues(tpool.workers(), std::numeric_limits<scalar_t>::max());

        loopit(nano::size(task.idims()), [&] (const tensor_size_t feature, const size_t t)
        {
                wlearner_linear_t learner;
                const auto value = learner.fit(task, fold, residuals, feature, type);
                if (value < tvalues[t])
                {
                        tvalues[t] = value;
                        std::swap(learners[t], learner);
                }
        });

        *this = learners[std::min_element(tvalues.begin(), tvalues.end()) - tvalues.begin()];
}

scalar_t wlearner_linear_t::fit(const task_t& task, const fold_t& fold, const tensor4d_t& residuals,
        const tensor_size_t feature, const wlearner_type /*type*/)
{
        scalar_t fvalues1 = 0, fvalues2 = 0;
        tensor3d_t residuals1(task.odims());
        tensor3d_t residuals2(task.odims());

        residuals1.zero();
        residuals2.zero();

        for (size_t i = 0, size = task.size(fold); i < size; ++ i)
        {
                const auto input = task.input(fold, i);
                const auto fvalue = input(feature);

                fvalues1 += fvalue;
                fvalues2 += fvalue * fvalue;
                residuals1.array() += residuals.array(i);
                residuals2.array() += fvalue * residuals.array(i);
        }

        m_a.resize(task.odims());
        m_b.resize(task.odims());

        const auto count = static_cast<scalar_t>(task.size(fold));
        m_a.array() =
                (residuals2.array() - residuals1.array() * fvalues1 / count) /
                (fvalues2 - nano::square(fvalues1));

        // todo: finish computing m_b
        // todo: finish computing the fitting value of this feature
        // todo: implement both discrete and real learners
        // todo: implement subsampling

        const scalar_t value = 0.0;

        return value;
}

bool wlearner_linear_t::save(obstream_t& stream) const
{
        return  stream.write(m_feature) &&
                stream.write_tensor(m_a) &&
                stream.write_tensor(m_b);
}

bool wlearner_linear_t::load(ibstream_t& stream)
{
        return  stream.read(m_feature) &&
                stream.read_tensor(m_a) &&
                stream.read_tensor(m_b);
}
