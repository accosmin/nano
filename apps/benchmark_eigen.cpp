#include "tensor.h"
#include "logger.h"
#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/random.h"
#include "tensor/numeric.h"
#include "text/table_row_mark.h"
#include <iostream>

namespace
{
        using namespace nano;

        auto rng_value = nano::make_rng(scalar_t(-1e-3), scalar_t(+1e-3));

        scalar_t make_scalar()
        {
                return rng_value();
        }

        vector_t make_vector(const tensor_size_t dims)
        {
                vector_t x(dims);
                nano::set_random(rng_value, x);
                return x;
        }

        matrix_t make_matrix(const tensor_size_t rows, const tensor_size_t cols)
        {
                matrix_t x(rows, cols);
                nano::set_random(rng_value, x);
                return x;
        }

        template <typename toperator>
        void store(table_row_t& row, const tensor_size_t flops, const toperator& op)
        {
                const auto trials = size_t(16);
                const auto duration = nano::measure_robustly_psec([&] () { op(); }, trials);
                row << std::chrono::duration_cast<microseconds_t>(duration).count() << nano::gflops(flops, duration);
        }

        void measure_level1(const tensor_size_t dims,
                table_row_t& row1, table_row_t& row2,
                table_row_t& row3, table_row_t& row4,
                table_row_t& row5, table_row_t& row6)
        {
                auto a = make_scalar();
                auto b = make_scalar();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row1, dims, [&] () { z = x.array() + c; });
                store(row2, dims, [&] () { z = x.array() + y.array(); });
                store(row3, 2 * dims, [&] () { z = x.array() * a + c; });
                store(row4, 2 * dims, [&] () { z = x.array() * a + y.array(); });
                store(row5, 3 * dims, [&] () { z = x.array() * a + y.array() * b; });
                store(row6, 4 * dims, [&] () { z = x.array() * a + y.array() * b + c; });
        }

        void measure_level2(const tensor_size_t dims,
                table_row_t& row1, table_row_t& row2, table_row_t& row3, table_row_t& row4)
        {
                auto A = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);
                auto c = make_scalar();

                store(row1, 2 * dims * dims, [&] () { z = A * x; });
                store(row2, 2 * dims * dims + dims, [&] () { z = (A * x).array() + c; });
                store(row3, 2 * dims * dims + dims, [&] () { z = A * x + y; });
                store(row4, 2 * dims * dims + dims, [&] () { Z = x * y.transpose() + C; });
        }

        auto measure_level3(const tensor_size_t dims,
                table_row_t& row1, table_row_t& row2)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                store(row1, 2 * dims * dims * dims, [&] () { Z = A * B + C; });
                store(row2, 2 * dims * dims * dims, [&] () { Z = A * B.transpose() + C; });
        }

        template <typename top>
        void foreach_dims(const tensor_size_t min, const tensor_size_t max, const top& op)
        {
                for (tensor_size_t dim = min; dim <= max; dim *= 2)
                {
                        op(dim);
                }
        }

        void fillheader(const tensor_size_t min, const tensor_size_t max, table_t& table)
        {
                foreach_dims(min, max, [&] (const tensor_size_t dims)
                {
                        const auto kilo = tensor_size_t(1) << 10;
                        const auto mega = tensor_size_t(1) << 20;
                        const auto value = (dims < kilo) ? dims : (dims < mega ? (dims / kilo) : (dims / mega));
                        const auto units = (dims < kilo) ? string_t("") : (dims < mega ? string_t("K") : string_t("M"));
                        const auto header = to_string(value) + units;
                        table.header() << (header + "[us]") << "gflop/s";
                });
        }
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark linear algebra operations using Eigen");
        cmdline.add("", "min-dims",     "minimum number of dimensions [16, 1024]", "16");
        cmdline.add("", "max-dims",     "maximum number of dimensions [16, 4096]", "1024");
        cmdline.add("", "level1",       "benchmark level1 operations (vector-vector)");
        cmdline.add("", "level2",       "benchmark level2 operations (matrix-vector)");
        cmdline.add("", "level3",       "benchmark level3 operations (matrix-matrix)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto min_dims = clamp(cmdline.get<tensor_size_t>("min-dims"), tensor_size_t(16), tensor_size_t(1024));
        const auto max_dims = clamp(cmdline.get<tensor_size_t>("max-dims"), min_dims, tensor_size_t(4096));
        const auto level1 = cmdline.has("level1");
        const auto level2 = cmdline.has("level2");
        const auto level3 = cmdline.has("level3");

        if (!level1 && !level2 && !level3)
        {
                cmdline.usage();
        }

        if (level1)
        {
                const auto min = 1024 * min_dims;
                const auto max = 1024 * max_dims;

                table_t table;
                table.header() << "operation";
                fillheader(min, max, table);
                {
                        auto& row1 = table.append() << "z = x + c";
                        auto& row2 = table.append() << "z = x + y";
                        auto& row3 = table.append() << "z = ax + c";
                        auto& row4 = table.append() << "z = ax + y";
                        auto& row5 = table.append() << "z = ax + by";
                        auto& row6 = table.append() << "z = ax + by + c";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level1(dims, row1, row2, row3, row4, row5, row6);
                        });
                }
                std::cout << table;
        }
        if (level2)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table;
                table.header() << "operation";
                fillheader(min, max, table);
                {
                        auto& row1 = table.append() << "z = Ax";
                        auto& row2 = table.append() << "z = Ax + c";
                        auto& row3 = table.append() << "z = Ax + y";
                        auto& row4 = table.append() << "Z = xy^t + C";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level2(dims, row1, row2, row3, row4);
                        });
                }
                std::cout << table;
        }
        if (level3)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table;
                table.header() << "operation";
                fillheader(min, max, table);
                {
                        auto& row1 = table.append() << "Z = AB + C";
                        auto& row2 = table.append() << "Z = AB^t + C";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level3(dims, row1, row2);
                        });
                }
                std::cout << table;
        }

        return EXIT_SUCCESS;
}

