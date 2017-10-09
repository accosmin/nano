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
        minibatch_t minibatch(end - begin, task.idims(), task.odims());
        for (size_t index = begin; index < end; ++ index)
        {
                const auto sample = task.get(fold, index);
                minibatch.m_inputs.vector(index - begin) = sample.m_input.vector();
                minibatch.m_targets.vector(index - begin) = sample.m_target.vector();
                minibatch.m_labels[index - begin] = sample.m_label;
        }

        return minibatch;
}
