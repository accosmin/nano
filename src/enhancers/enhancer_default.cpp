#include "enhancer_default.h"

using namespace nano;

json_reader_t& enhancer_default_t::config(json_reader_t& reader)
{
        return reader;
}

json_writer_t& enhancer_default_t::config(json_writer_t& writer) const
{
        return writer;
}

minibatch_t enhancer_default_t::get(const task_t& task, const fold_t& fold, const size_t begin, const size_t end) const
{
        return task.get(fold, begin, end);
}
