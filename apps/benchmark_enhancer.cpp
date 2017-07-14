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
        cmdline_t cmdline("benchmark task enhancers");
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
        table.header() << "" << "" << "" << "";
        for (const auto& id : get_enhancers().ids())
        {
                table.header() << ("it(" + id + ")");
        }
        {
                auto& row = table.append() << "#samples" << "isize" << "init[ms]" << "shuffle[us]";
                for (const auto& id : get_enhancers().ids())
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
                row << measure<milliseconds_t>([&] () { task->load(); }, 1).count();
                row << measure<milliseconds_t>([&] () { task->shuffle({0, protocol::train}); }, 16).count();

                // vary the enhancer
                for (const auto& id : get_enhancers().ids())
                {
                        const auto enhancer = get_enhancers().get(id);

                        const auto fold = fold_t{0, protocol::train};
                        const auto fold_size = task->size(fold);

                        const auto duration = measure<microseconds_t>([&] ()
                        {
                                volatile long double sum = 0;
                                for (size_t index = 0; index < fold_size; ++ index)
                                {
                                        const auto sample = enhancer->get(*task, fold, index);
                                        const auto& input = sample.m_input;
                                        const auto& target = sample.m_target;
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
