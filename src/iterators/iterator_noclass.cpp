#include "math/random.h"
#include "tensor/numeric.h"
#include "iterator_noclass.h"

using namespace nano;

iterator_noclass_t::iterator_noclass_t(const string_t& config) :
        iterator_t(to_params(config, "ratio", "0.1[0,1]", "noise", "0.1[0,1]"))
{
}

sample_t iterator_noclass_t::get(const task_t& task, const fold_t& fold, const size_t index) const
{
        sample_t sample = task.get(fold, index);

        const auto ratio = from_params<scalar_t>(config(), "ratio");
        const auto noise = from_params<scalar_t>(config(), "noise");

        auto rng = make_rng<scalar_t>(-1, +1);
        if (std::fabs(rng()) < ratio)
        {
                // replace input with random [0, 1] values with no class label
                nano::set_random([&] () { return (rng() + 1) / 2; }, sample.m_input);
                sample.m_target.constant(neg_target());
        }
        else
        {
                // keep input and add some salt & pepper noise
                nano::add_random([&] () { return rng() * noise; }, sample.m_input);
        }

        return sample;
}