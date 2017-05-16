#include "measure.h"
#include "iterator.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "tensor/numeric.h"
#include <iostream>

using namespace nano;

namespace nano
{
        template <typename tvalue>
        string_t serialize_to_string(const tvalue value)
        {
                std::stringstream s;
                s << value;
                return s.str();
        }

        template <> string_t to_string<tensor2d_dims_t>(const tensor2d_dims_t dims) { return serialize_to_string(dims); }
        template <> string_t to_string<tensor3d_dims_t>(const tensor3d_dims_t dims) { return serialize_to_string(dims); }
        template <> string_t to_string<tensor4d_dims_t>(const tensor4d_dims_t dims) { return serialize_to_string(dims); }
}

int main(int argc, const char *argv[])
{
        const size_t kilo = 1000;

        // parse the command line
        cmdline_t cmdline("benchmark task iterators");
        cmdline.add("", "min-samples",  "minimum number of samples [1K, 10K]", 10 * 1000);
        cmdline.add("", "max-samples",  "maximum number of samples [10K, 1M]", 80 * 1000);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_tasksize = clamp(cmdline.get<size_t>("min-samples"), kilo, 10 * kilo);
        const auto max_tasksize = clamp(cmdline.get<size_t>("max-samples"), min_tasksize, 1000 * kilo);

        NANO_UNUSED1(argc);
        NANO_UNUSED1(argv);

        const auto cmd_rows = 32;
        const auto cmd_cols = 32;
        const auto cmd_color = color_mode::rgb;

        // prepare table
        table_t table;
        table.header() << "" << "" << "";
        for (const auto& id : get_iterators().ids())
        {
                table.header() << ("it(" + id + ")");
        }
        {
                auto& row = table.append() << "#samples" << "isize" << "init[ms]";
                for (const auto& id : get_iterators().ids())
                {
                        NANO_UNUSED1(id);
                        row << "[us/sample]";
                }
                table.append(table_row_t::storage::delim);
        }

        // vary the task size
        for (size_t task_size = min_tasksize; task_size <= max_tasksize; task_size *= 2)
        {
                // measure task generation
                auto task = get_tasks().get("synth-charset", to_params(
                        "type", charset_type::digit, "color", cmd_color,
                        "irows", cmd_rows, "icols", cmd_cols, "count", task_size));

                auto& row = table.append();
                row << task_size << task->idims();
                {
                        const auto duration = measure_robustly<milliseconds_t>([&] ()
                        {
                                task->load();
                        }, 1);
                        row << duration.count();
                }

                // vary the iterator
                for (const auto& id : get_iterators().ids())
                {
                        const auto iterator = get_iterators().get(id);

                        const auto fold = fold_t{0, protocol::train};
                        const auto fold_size = task->size(fold);

                        const auto duration = measure_robustly<microseconds_t>([&] ()
                        {
                                volatile long double sum = 0;
                                for (size_t index = 0; index < fold_size; ++ index)
                                {
                                        const auto input = iterator->input(*task, fold, index);
                                        const auto target = iterator->target(*task, fold, index);
                                        sum += input.vector().sum() - target.vector().sum();
                                }
                        }, 1);
                        row << idiv(duration.count(), fold_size);
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
