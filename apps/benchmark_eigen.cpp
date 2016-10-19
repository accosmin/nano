#include "tensor.h"
#include "logger.h"
#include "measure.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/random.h"
#include "tensor/numeric.h"
#include "text/table_row_mark.h"
#ifdef NANO_WITH_OPENCL
#include "opencl/ocl.h"
#include "math/epsilon.h"
#endif
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

#ifdef NANO_WITH_OPENCL
        static void store(table_row_t& row, const picoseconds_t duration)
        {
                row << std::chrono::duration_cast<microseconds_t>(duration).count() << " - ";
        }

        static void assert_equal(const vector_t& v1, const vector_t& v2)
        {
                if ((v1 - v2).template lpNorm<Eigen::Infinity>() > nano::epsilon0<scalar_t>())
                {
                        throw std::runtime_error("invalid OpenCL kernel");
                }
        }

        static void assert_equal(const matrix_t& m1, const matrix_t& m2)
        {
                if ((m1 - m2).template lpNorm<Eigen::Infinity>() > nano::epsilon0<scalar_t>())
                {
                        throw std::runtime_error("invalid OpenCL kernel");
                }
        }

        static auto measure_level1_kernel(const cl::Kernel& kernel, const tensor_size_t dims1)
        {
                return measure_robustly_psec([&] () { ocl::wait(ocl::level1::enqueue(kernel, dims1)); }, trials);
        }

        static auto measure_level2_kernel(const cl::Kernel& kernel, const tensor_size_t dims1)
        {
                return measure_robustly_psec([&] () { ocl::wait(ocl::level2::enqueue(kernel, dims1)); }, trials);
        }

        static auto measure_level3_kernel(const cl::Kernel& kernel, const tensor_size_t dims1, const tensor_size_t dims2)
        {
                return measure_robustly_psec([&] () { ocl::wait(ocl::level3::enqueue(kernel, dims1, dims2)); }, trials);
        }

        void measure_level1_opencl(const tensor_size_t dims,
                table_row_t& row_read, table_row_t& row_write,
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

                cl::Buffer xbuffer = ocl::make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = ocl::make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = ocl::make_buffer(z, CL_MEM_READ_ONLY);

                ocl::write(xbuffer, x);
                ocl::write(ybuffer, y);

                {
                        store(row_read, nano::measure_robustly_psec([&] ()
                        {
                                ocl::read(xbuffer, z);
                        }, trials));
                        assert_equal(z, x);
                }
                {
                        store(row_write, nano::measure_robustly_psec([&] ()
                        {
                                ocl::write(xbuffer, z);
                        }, trials));
                        assert_equal(z, x);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vpc");
                        ocl::set_args(kernel, xbuffer, c, zbuffer, dims);
                        store(row_vpc, dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x.array() + c);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vpv");
                        ocl::set_args(kernel, xbuffer, ybuffer, zbuffer, dims);
                        store(row_vpv, dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x + y);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpc");
                        ocl::set_args(kernel, xbuffer, a, c, zbuffer, dims);
                        store(row_vcpc, 2 * dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x.array() * a + c);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpv");
                        ocl::set_args(kernel, xbuffer, a, ybuffer, zbuffer, dims);
                        store(row_vcpv, 2 * dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x * a + y);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpvc");
                        ocl::set_args(kernel, xbuffer, a, ybuffer, b, zbuffer, dims);
                        store(row_vcpvc, 3 * dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x * a + y * b);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("vcpvcpc");
                        ocl::set_args(kernel, xbuffer, a, ybuffer, b, c, zbuffer, dims);
                        store(row_vcpvcpc, 4 * dims, measure_level1_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, x.array() * a + y.array() * b + c);
                }
        }
#endif

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

#ifdef NANO_WITH_OPENCL
        auto measure_level2_opencl(const tensor_size_t dims,
                table_row_t& row_mv, table_row_t& row_mvpc, table_row_t& row_mvpv)
        {
                auto A = make_matrix(dims, dims);
                auto x = make_vector(dims);
                auto y = make_vector(dims);
                auto z = make_vector(dims);
                auto c = make_scalar();

                cl::Buffer Abuffer = ocl::make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer xbuffer = ocl::make_buffer(x, CL_MEM_READ_WRITE);
                cl::Buffer ybuffer = ocl::make_buffer(y, CL_MEM_READ_WRITE);
                cl::Buffer zbuffer = ocl::make_buffer(z, CL_MEM_READ_ONLY);

                ocl::write(Abuffer, A);
                ocl::write(xbuffer, x);
                ocl::write(ybuffer, y);

                {
                        cl::Kernel kernel = ocl::make_kernel("mv");
                        ocl::set_args(kernel, Abuffer, xbuffer, zbuffer, dims, dims);
                        store(row_mv, 2 * dims * dims, measure_level2_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, A * x);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("mvpc");
                        ocl::set_args(kernel, Abuffer, xbuffer, c, zbuffer, dims, dims);
                        store(row_mvpc, 2 * dims * dims + dims, measure_level2_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, (A * x).array() + c);
                }
                {
                        cl::Kernel kernel = ocl::make_kernel("mvpv");
                        ocl::set_args(kernel, Abuffer, xbuffer, ybuffer, zbuffer, dims, dims);
                        store(row_mvpv, 2 * dims * dims + dims, measure_level2_kernel(kernel, dims));
                        ocl::read(zbuffer, z);
                        assert_equal(z, (A * x + y).array());
                }
        }
#endif

        auto measure_level3(const tensor_size_t dims,
                table_row_t& row_mm)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                {
                        store(row_mm, 2 * dims * dims * dims, nano::measure_robustly_psec([&] ()
                        {
                                Z = A * B;
                        }, trials));
                }
        }

#ifdef NANO_WITH_OPENCL
        auto measure_level3_opencl(const tensor_size_t dims,
                table_row_t& row_mm)
        {
                auto A = make_matrix(dims, dims);
                auto B = make_matrix(dims, dims);
                auto Z = make_matrix(dims, dims);

                cl::Buffer Abuffer = ocl::make_buffer(A, CL_MEM_READ_WRITE);
                cl::Buffer Bbuffer = ocl::make_buffer(B, CL_MEM_READ_WRITE);
                cl::Buffer Zbuffer = ocl::make_buffer(Z, CL_MEM_READ_ONLY);

                ocl::write(Abuffer, A);
                ocl::write(Bbuffer, B);

                {
                        cl::Kernel kernel = ocl::make_kernel("mm");
                        ocl::set_args(kernel, Abuffer, Bbuffer, Zbuffer, dims, dims, dims);
                        store(row_mm, 2 * dims * dims * dims, measure_level3_kernel(kernel, dims, dims));
                        ocl::read(Zbuffer, Z);
                        assert_equal(Z, A * B);
                }
        }
#endif
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark linear algebra operations using Eigen and OpenCL (if available)");
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

#ifdef NANO_WITH_OPENCL
        try
        {
        // initialize OpenCL context
        ocl::select(CL_DEVICE_TYPE_GPU);
#endif

        const auto foreach_dims = [&] (const auto min, const auto max, const auto& op)
        {
                for (tensor_size_t dims = min; dims <= max; dims *= 2)
                {
                        op(dims);
                }
        };
        const auto fillheader = [&] (const auto min, const auto max, table_t& table)
        {
                table.header() << "device";
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
                        auto& row_vpc = table.append("z = x + c") << "CPU";
                        auto& row_vpv = table.append("z = x + y") << "CPU";
                        auto& row_vcpc = table.append("z = ax + c") << "CPU";
                        auto& row_vcpv = table.append("z = ax + y") << "CPU";
                        auto& row_vcpvc = table.append("z = ax + by") << "CPU";
                        auto& row_vcpvcpc = table.append("z = ax + by + c") << "CPU";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level1(dims, row_vpc, row_vpv, row_vcpc, row_vcpv, row_vcpvc, row_vcpvcpc);
                        });
                }
#ifdef NANO_WITH_OPENCL
                {
                        auto& row_read = table.append("read") << "OpenCL";
                        auto& row_write = table.append("write") << "OpenCL";
                        auto& row_vpc = table.append("z = x + c") << "OpenCL";
                        auto& row_vpv = table.append("z = x + y") << "OpenCL";
                        auto& row_vcpc = table.append("z = ax + c") << "OpenCL";
                        auto& row_vcpv = table.append("z = ax + y") << "OpenCL";
                        auto& row_vcpvc = table.append("z = ax + by") << "OpenCL";
                        auto& row_vcpvcpc = table.append("z = ax + by + c") << "OpenCL";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level1_opencl(dims, row_read, row_write, row_vpc, row_vpv, row_vcpc, row_vcpv, row_vcpvc, row_vcpvcpc);
                        });
                }
#endif
                std::cout << table;
        }
        if (level2)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table("operation");
                fillheader(min, max, table);
                {
                        auto& row_mv = table.append("z = Ax") << "CPU";
                        auto& row_mvpc = table.append("z = Ax + c") << "CPU";
                        auto& row_mvpv = table.append("z = Ax + y") << "CPU";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level2(dims, row_mv, row_mvpc, row_mvpv);
                        });
                }
#ifdef NANO_WITH_OPENCL
                {
                        auto& row_mv = table.append("z = Ax") << "OpenCL";
                        auto& row_mvpc = table.append("z = Ax + c") << "OpenCL";
                        auto& row_mvpv = table.append("z = Ax + y") << "OpenCL";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level2_opencl(dims, row_mv, row_mvpc, row_mvpv);
                        });
                }
#endif
                std::cout << table;
        }
        if (level3)
        {
                const auto min = min_dims;
                const auto max = max_dims;

                table_t table("operation");
                fillheader(min, max, table);
                {
                        auto& row_mm = table.append("Z = AB") << "CPU";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level3(dims, row_mm);
                        });
                }
#ifdef NANO_WITH_OPENCL
                {
                        auto& row_mm = table.append("Z = AB") << "OpenCL";
                        foreach_dims(min, max, [&] (const auto dims)
                        {
                                measure_level3_opencl(dims, row_mm);
                        });
                }
#endif
                std::cout << table;
        }

#ifdef NANO_WITH_OPENCL
        }
        catch (cl::Error& e)
        {
                log_error() << "OpenCL fatal error: " << e.what() << " (" << ocl::error_string(e.err()) << ")!";
                return EXIT_FAILURE;
        }
        catch (std::exception& e)
        {
                log_error() << "OpenCL fatal error: " << e.what() << "!";
                return EXIT_FAILURE;
        }
#endif

        return EXIT_SUCCESS;
}

