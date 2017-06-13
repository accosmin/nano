#include "iterator_default.h"

namespace nano
{
        iterator_default_t::iterator_default_t(const string_t& config) :
                iterator_t(config)
        {
        }

        sample_t iterator_default_t::get(const task_t& task, const fold_t& fold, const size_t index) const
        {
                return task.get(fold, index);
        }
}
