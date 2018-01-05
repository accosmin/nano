#include "enhancer.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/numeric.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "chrono/measure.h"
#include "tensor/numeric.h"
#include <iostream>

using namespace nano;

int main(int argc, const char *argv[])
{
        const size_t kilo = 1000;

        // parse the command line
        cmdline_t cmdline("benchmark task enhancers");
        cmdline.add("", "min-samples",  "minimum number of samples [1K, 10K]", 10 * 1000);
        cmdline.add("", "max-samples",  "maximum number of samples [10K, 1M]", 80 * 1000);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_tasksize = clamp(cmdline.get<size_t>("min-samples"), kilo, 10 * kilo);
        const auto max_tasksize = clamp(cmdline.get<size_t>("max-samples"), min_tasksize, 1000 * kilo);

        const auto cmd_rows = 32;
        const auto cmd_cols = 32;
        const auto cmd_color = color_mode::rgb;
        const auto min_minibatch = size_t(1);
        const auto max_minibatch = size_t(16);

        // prepare table
        table_t table;
        table.header()
                << colspan(5) << ""
                << colspan(get_enhancers().size()) << colfill('=') << alignment::center << "enhancers[us/sample]";
        table.delim();
        table.append()
                << "#samples" << "isize" << "batch" << "init[ms]" << "shuffle[us]"
                << get_enhancers().ids();
        table.delim();

        // vary the task size
        for (size_t task_size = min_tasksize; task_size <= max_tasksize; task_size *= 2)
        {
                // measure task generation
                auto task = get_tasks().get("synth-charset");
                task->config(json_writer_t().object(
                        "type", charset_type::digit, "color", cmd_color,
                        "irows", cmd_rows, "icols", cmd_cols, "count", task_size).str());

                // vary the minibatch size
                for (size_t minibatch = min_minibatch; minibatch <= max_minibatch; minibatch *= 2)
                {
                        auto& row = table.append();
                        row << task_size << task->idims() << ("x" + to_string(minibatch));
                        row << measure<milliseconds_t>([&] () { task->load(); }, 1).count();
                        row << measure<microseconds_t>([&] () { task->shuffle({0, protocol::train}); }, 16).count();

                        const auto fold = fold_t{0, protocol::train};
                        const auto fold_size = task->size(fold);

                        // vary the enhancer
                        for (const auto& id : get_enhancers().ids())
                        {
                                const auto enhancer = get_enhancers().get(id);

                                const auto duration = measure<microseconds_t>([&] ()
                                {
                                        volatile long double sum = 0;
                                        for (size_t index = 0; index + minibatch < fold_size; index += minibatch)
                                        {
                                                const auto mbatch = enhancer->get(*task, fold, index, index + minibatch);
                                                const auto& idata = mbatch.idata();
                                                const auto& odata = mbatch.odata();
                                                sum += idata.vector().sum() - odata.vector().sum();
                                        }
                                }, 1);
                                row << idiv(duration.count(), fold_size);
                        }
                }

                if (task_size * 2 <= max_tasksize)
                {
                        table.delim();
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
