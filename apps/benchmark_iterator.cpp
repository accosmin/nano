#include "nano.h"
#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "text/to_params.h"
#include "tensor/numeric.h"
#include "measure_and_log.h"
#include <iostream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        const size_t kilo = 1000;
        const size_t threads = nano::logical_cpus();

        // parse the command line
        cmdline_t cmdline("benchmark batch optimizers");
        cmdline.add("", "min-samples",  "minimum number of samples [1K, 10K]", 10 * 1000);
        cmdline.add("", "max-samples",  "maximum number of samples [10K, 1M]", 80 * 1000);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_tasksize = clamp(cmdline.get<size_t>("min-samples"), kilo, 10 * kilo);
        const auto max_tasksize = clamp(cmdline.get<size_t>("max-samples"), min_tasksize, 1000 * kilo);
        const auto min_batchsize = 8 * threads;
        const auto max_batchsize = 256 * threads;

        NANO_UNUSED1(argc);
        NANO_UNUSED1(argv);

        const auto cmd_rows = 32;
        const auto cmd_cols = 32;
        const auto cmd_color = color_mode::rgba;

        // prepare table
        nano::table_t table;
        table.header() << "#samples" << "init [ms]";
        for (size_t batch_size = min_batchsize; batch_size <= max_batchsize; batch_size *= 2)
        {
                table.header() << ("batch " + to_string(batch_size) + " [ms]");
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
                        const auto duration = nano::measure_robustly<milliseconds_t>([&] ()
                        {
                                task->load();
                        }, 1);
                        row << duration.count();
                }

                // vary the minibatch size
                for (size_t batch_size = min_batchsize; batch_size <= max_batchsize; batch_size *= 2)
                {
                        const auto duration = nano::measure_robustly<milliseconds_t>([&] ()
                        {
                                const auto fold = fold_t{0, protocol::train};
                                const auto epochs = 100 * nano::idiv(task->size(fold), batch_size);

                                auto iterator = get_iterators().get("default");
                                iterator->reset(*task, fold, batch_size);

                                volatile size_t count = 0;
                                volatile long double sum = 0;
                                for (size_t epoch = 0; epoch < epochs; ++ epoch)
                                {
                                        count += it->size();
                                        for (size_t index = it->begin(), end = it->end(); index < end; ++ index)
                                        {
                                                const auto target = task->target(fold, index);
                                                sum += target.vector().sum();
                                        }
                                        it->next();
                                }
                        }, 1);
                        row << duration.count();
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
