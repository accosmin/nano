#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "task_iterator.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "tensor/numeric.h"
#include "measure_and_log.h"
#include "tasks/task_charset.h"
#include "text/table_row_mark.h"
#include <iostream>

int main(int argc, const char *argv[])
{
        using namespace nano;

        NANO_UNUSED1(argc);
        NANO_UNUSED1(argv);

        const tensor_size_t cmd_rows = 32;
        const tensor_size_t cmd_cols = 32;
        const color_mode cmd_color = color_mode::rgba;

        const size_t kilo = 1000;
        const size_t threads = nano::logical_cpus();

        const sizes_t task_sizes = { 10 * kilo, 20 * kilo, 40 * kilo, 80 * kilo };
        const sizes_t batch_sizes = { 8 * threads, 16 * threads, 32 * threads, 64 * threads, 128 * threads, 256 * threads };

        // prepare table
        nano::table_t table;
        table.header() << "#samples" << "init [ms]";
        for (const size_t batch_size : batch_sizes)
        {
                table.header() << ("batch " + to_string(batch_size) + " [ms]");
        }

        // vary the task size
        for (const size_t task_size : task_sizes)
        {
                auto& row = table.append();
                row << task_size;

                // measure task generation
                charset_task_t task(to_params(
                        "type", charset_mode::digit, "color", cmd_color,
                        "irows", cmd_rows, "icols", cmd_cols, "count", task_size));
                {
                        const auto duration = nano::measure_robustly<milliseconds_t>([&] ()
                        {
                                task.load();
                        }, 1);
                        row << duration.count();
                }

                // vary the minibatch size
                for (const size_t batch_size : batch_sizes)
                {
                        // vary the minibatch selection
                        const auto duration = nano::measure_robustly<milliseconds_t>([&] ()
                        {
                                const auto fold = fold_t{0, protocol::train};
                                const auto epochs = 100 * nano::idiv(task.size(fold), batch_size);

                                task_iterator_t it(task, fold, batch_size);

                                volatile size_t count = 0;
                                volatile long double sum = 0;
                                for (size_t epoch = 0; epoch < epochs; ++ epoch)
                                {
                                        count += it.end() - it.begin();
                                        for (size_t index = it.begin(), end = it.end(); index < end; ++ index)
                                        {
                                                const auto target = task.target(fold, index);
                                                sum += target.vector().sum();
                                        }
                                        it.next();
                                }
                        }, 1);
                        row << duration.count();
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
