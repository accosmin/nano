#include "enhancer_default.h"

using namespace nano;

enhancer_default_t::enhancer_default_t(const string_t& config) :
        enhancer_t(config)
{
}

sample_t enhancer_default_t::get(const task_t& task, const fold_t& fold, const size_t index) const
{
        return task.get(fold, index);
}

minibatch_t enhancer_default_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        return task.get(fold, begin, end);
}
