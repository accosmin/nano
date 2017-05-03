#include "sampler_none.h"

namespace nano
{
        sampler_none_t::sampler_none_t(const string_t& config) :
                sampler_t(config)
        {
        }

        tensor3d_t sampler_none_t::input(const task_t& task, const fold_t& fold, const size_t index)
        {
                return task.input(fold, index);
        }

        tensor3d_t sampler_none_t::target(const task_t& task, const fold_t& fold, const size_t index)
        {
                return task.target(fold, index);
        }
}
