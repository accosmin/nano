#include "nano.h"
#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "text/to_params.h"
#include "tensor/numeric.h"
#include <iostream>

int main(int argc, const char *argv[])
{
        using namespace nano;

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
        const auto cmd_color = color_mode::rgba;

        // prepare table
        table_t table;
        table.header() << "#samples" << "init [ms]";
        for (const auto& id : get_iterators().ids())
        {
                table.header() << ("it(" + id + ") [ms]");
        }

        // vary the task size
        for (size_t task_size = min_tasksize; task_size <= max_tasksize; task_size *= 2)
        {
                auto& row = table.append();
                row << task_size;

                // measure task generation
                auto task = get_tasks().get("synth-charset", to_params(
                        "type", charset_type::digit, "color", cmd_color,
                        "irows", cmd_rows, "icols", cmd_cols, "count", task_size));
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

                        const auto duration = measure_robustly<milliseconds_t>([&] ()
                        {
                                const auto fold = fold_t{0, protocol::train};
                                const auto fold_size = task->size(fold);
                                const auto epochs = 100;

                                volatile long double sum = 0;
                                for (size_t epoch = 0; epoch < epochs; ++ epoch)
                                {
                                        for (size_t index = 0; index < fold_size; ++ index)
                                        {
                                                const auto input = iterator->input(*task, fold, index);
                                                const auto target = iterator->target(*task, fold, index);
                                                sum += input.vector().sum() - target.vector().sum();
                                        }
                                }
                        }, 1);
                        row << duration.count();
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
