#include "tensor.h"
#include "text/table.h"
#include "math/random.h"
#include "text/cmdline.h"
#include "chrono/measure.h"
#include "tensor/numeric.h"
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
        void store(row_t& row, const tensor_size_t flops, const toperator& op)
        {
                const auto trials = size_t(16);
                const auto duration = nano::measure<picoseconds_t>([&] () { op(); }, trials);
                row << nano::gflops(flops, duration);
        }

        void measure_level11(const tensor_size_t dims, row_t& row)
        {
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                store(row, dims, [&] () { z = x.array() + c; });
        }

        void measure_level12(const tensor_size_t dims, row_t& row)
        {
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row, dims, [&] () { z = x.array() + y.array(); });
        }

        void measure_level13(const tensor_size_t dims, row_t& row)
        {
                auto a = make_scalar();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 2 * dims, [&] () { z = x.array() * a + c; });
        }

        void measure_level14(const tensor_size_t dims, row_t& row)
        {
                auto a = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 2 * dims, [&] () { z = x.array() * a + y.array(); });
        }

        void measure_level15(const tensor_size_t dims, row_t& row)
        {
                auto a = make_scalar();
                auto b = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 3 * dims, [&] () { z = x.array() * a + y.array() * b; });
        }

        void measure_level16(const tensor_size_t dims, row_t& row)
        {
                auto a = make_scalar();
                auto b = make_scalar();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 4 * dims, [&] () { z = x.array() * a + y.array() * b + c; });
        }

        void measure_level21(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 2 * dims * dims, [&] () { z = A * x; });
        }

        void measure_level22(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto z = make_vector(dims);
                auto c = make_scalar();

                store(row, 2 * dims * dims + dims, [&] () { z = (A * x).array() + c; });
        }

        void measure_level23(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                store(row, 2 * dims * dims + dims, [&] () { z = A * x + y; });
        }

        void measure_level24(const tensor_size_t dims, row_t& row)
        {
                auto Z = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);

                store(row, 2 * dims * dims + dims, [&] () { Z.noalias() = x * y.transpose() + C; });
        }

        auto measure_level31(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                store(row, 2 * dims * dims * dims, [&] () { Z.noalias() = A * B; });
        }

        auto measure_level32(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                store(row, 2 * dims * dims * dims + dims * dims, [&] () { Z.noalias() = A * B + C; });
        }

        auto measure_level33(const tensor_size_t dims, row_t& row)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                store(row, 2 * dims * dims * dims + dims * dims, [&] () { Z.noalias() = A * B.transpose() + C; });
        }

        template <typename top>
        void foreach_dims(const tensor_size_t min, const tensor_size_t max, const top& op)
        {
                for (tensor_size_t dim = min; dim <= max; dim *= 2)
                {
                        op(dim);
                }
        }

        template <typename top>
        void foreach_dims_row(const tensor_size_t min, const tensor_size_t max, row_t& row, const top& op)
        {
                for (tensor_size_t dim = min; dim <= max; dim *= 2)
                {
                        op(dim, row);
                }
        }

        void fillheader(const tensor_size_t min, const tensor_size_t max, table_t& table)
        {
                auto& header = table.header();
                header << "operation";
                foreach_dims(min, max, [&] (const tensor_size_t dims)
                {
                        const auto kilo = tensor_size_t(1) << 10;
                        const auto mega = tensor_size_t(1) << 20;
                        const auto value = (dims < kilo) ? dims : (dims < mega ? (dims / kilo) : (dims / mega));
                        const auto units = (dims < kilo) ? string_t("") : (dims < mega ? string_t("K") : string_t("M"));
                        header << (to_string(value) + units + "[gflop/s]");
                });
                table.delim();
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
                fillheader(min, max, table);
                {
                        foreach_dims_row(min, max, table.append() << "z = x + c", measure_level11);
                        foreach_dims_row(min, max, table.append() << "z = x + y", measure_level12);
                        foreach_dims_row(min, max, table.append() << "z = ax + c", measure_level13);
                        foreach_dims_row(min, max, table.append() << "z = ax + y", measure_level14);
                        foreach_dims_row(min, max, table.append() << "z = ax + by", measure_level15);
                        foreach_dims_row(min, max, table.append() << "z = ax + by + c", measure_level16);
                }
                std::cout << table;
        }
        if (level2)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table;
                fillheader(min, max, table);
                {
                        foreach_dims_row(min, max, table.append() << "z = Ax", measure_level21);
                        foreach_dims_row(min, max, table.append() << "z = Ax + c", measure_level22);
                        foreach_dims_row(min, max, table.append() << "z = Ax + y", measure_level23);
                        foreach_dims_row(min, max, table.append() << "Z = xy^t + C", measure_level24);
                }
                std::cout << table;
        }
        if (level3)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table;
                fillheader(min, max, table);
                {
                        foreach_dims_row(min, max, table.append() << "Z = AB", measure_level31);
                        foreach_dims_row(min, max, table.append() << "Z = AB + C", measure_level32);
                        foreach_dims_row(min, max, table.append() << "Z = AB^t + C", measure_level33);
                }
                std::cout << table;
        }

        return EXIT_SUCCESS;
}
