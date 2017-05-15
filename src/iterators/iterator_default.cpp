#include "iterator_default.h"

namespace nano
{
        iterator_default_t::iterator_default_t(const string_t& config) :
                iterator_t(config)
        {
        }

        tensor3d_t iterator_default_t::input(const task_t& task, const fold_t& fold, const size_t index) const
        {
                return task.input(fold, index);
        }

        tensor3d_t iterator_default_t::target(const task_t& task, const fold_t& fold, const size_t index) const
        {
                return task.target(fold, index);
        }
}
