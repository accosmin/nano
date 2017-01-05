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
        const size_t trials = 16;

        scalar_t make_scalar()
        {
                return rng_value();
        }

        vector_t make_vector(const tensor_size_t dims)
        {
                vector_t x(dims);
                tensor::set_random(rng_value, x);
                return x;
        }

        matrix_t make_matrix(const tensor_size_t rows, const tensor_size_t cols)
        {
                matrix_t x(rows, cols);
                tensor::set_random(rng_value, x);
                return x;
        }

        void store(table_row_t& row, const tensor_size_t flops, const picoseconds_t duration)
        {
                row << std::chrono::duration_cast<microseconds_t>(duration).count() << nano::gflops(flops, duration);
        }

        void measure_level1(const tensor_size_t dims,
                table_row_t& row_vpc, table_row_t& row_vpv,
                table_row_t& row_vcpc, table_row_t& row_vcpv,
                table_row_t& row_vcpvc, table_row_t& row_vcpvcpc)
        {
                auto a = make_scalar();
                auto b = make_scalar();
                auto c = make_scalar();
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);

                {
                        store(row_vpc, dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() + c;
                        }, trials));
                }
                {
                        store(row_vpv, dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() + y.array();
                        }, trials));
                }
                {
                        store(row_vcpc, 2 * dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() * a + c;
                        }, trials));
                }
                {
                        store(row_vcpv, 2 * dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() * a + y.array();
                        }, trials));
                }
                {
                        store(row_vcpvc, 3 * dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() * a + y.array() * b;
                        }, trials));
                }
                {
                        store(row_vcpvcpc, 4 * dims, nano::measure_robustly_psec([&] ()
                        {
                                z = x.array() * a + y.array() * b + c;
                        }, trials));
                }
        }

        void measure_level2(const tensor_size_t dims,
                table_row_t& row_mv, table_row_t& row_mvpc, table_row_t& row_mvpv)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);
                auto c = make_scalar();

                {
                        store(row_mv, 2 * dims * dims, nano::measure_robustly_psec([&] ()
                        {
                                z = A * x;
                        }, trials));
                }
                {
                        store(row_mvpc, 2 * dims * dims + dims, nano::measure_robustly_psec([&] ()
                        {
                                z = (A * x).array() + c;
                        }, trials));
                }
                {
                        store(row_mvpv, 2 * dims * dims + dims, nano::measure_robustly_psec([&] ()
                        {
                                z = A * x + y;
                        }, trials));
                }
        }

        auto measure_level3(const tensor_size_t dims,
                table_row_t& row_mmpc)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto C = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                {
                        store(row_mmpc, 2 * dims * dims * dims, nano::measure_robustly_psec([&] ()
                        {
                                Z = A * B + C;
                        }, trials));
                }
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

        const auto foreach_dims = [&] (const auto min, const auto max, const auto& op)
        {
                for (tensor_size_t dims = min; dims <= max; dims *= 2)
                {
                        op(dims);
                }
        };
        const auto fillheader = [&] (const auto min, const auto max, table_t& table)
        {
                foreach_dims(min, max, [&] (const tensor_size_t dims)
                {
                        const auto kilo = tensor_size_t(1) << 10;
                        const auto mega = tensor_size_t(1) << 20;
                        const auto value = (dims < kilo) ? dims : (dims < mega ? (dims / kilo) : (dims / mega));
                        const auto units = (dims < kilo) ? string_t("") : (dims < mega ? string_t("K") : string_t("M"));
                        const auto header = to_string(value) + units;
                        table.header() << (header + "[us]") << "GFLOPS";
                });
        };

        if (level1)
        {
                const auto min = 1024 * min_dims;
                const auto max = 1024 * max_dims;

                table_t table("operation");
                fillheader(min, max, table);
                {
                        auto& row_vpc = table.append("z = x + c");
                        auto& row_vpv = table.append("z = x + y");
                        auto& row_vcpc = table.append("z = ax + c");
                        auto& row_vcpv = table.append("z = ax + y");
                        auto& row_vcpvc = table.append("z = ax + by");
                        auto& row_vcpvcpc = table.append("z = ax + by + c");
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level1(dims, row_vpc, row_vpv, row_vcpc, row_vcpv, row_vcpvc, row_vcpvcpc);
                        });
                }
                std::cout << table;
        }
        if (level2)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table("operation");
                fillheader(min, max, table);
                {
                        auto& row_mv = table.append("z = Ax");
                        auto& row_mvpc = table.append("z = Ax + c");
                        auto& row_mvpv = table.append("z = Ax + y");
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level2(dims, row_mv, row_mvpc, row_mvpv);
                        });
                }
                std::cout << table;
        }
        if (level3)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table("operation");
                fillheader(min, max, table);
                {
                        auto& row_mmpc = table.append("Z = AB + C");
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level3(dims, row_mmpc);
                        });
                }
                std::cout << table;
        }

        return EXIT_SUCCESS;
}

