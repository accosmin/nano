#include "cortex.h"
#include "logger.h"
#include "text/table.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "chrono/measure.h"
#include "layers/affine3d.h"
#include "layers/affine4d.h"
#include <iostream>

using namespace nano;

namespace
{
        const auto trials = size_t(16);

        template <typename top, typename tidata, typename twdata, typename tbdata, typename todata>
        auto measure_output(const top& op, const tidata& idata, const twdata& wdata, const tbdata& bdata, todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.output(idata, wdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_output() * count, duration);
        }

        template <typename top, typename tidata, typename twdata, typename tbdata, typename todata>
        auto measure_ginput(const top& op, tidata& idata, const twdata& wdata, const tbdata& bdata, const todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.ginput(idata, wdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_ginput() * count, duration);
        }

        template <typename top, typename tidata, typename twdata, typename tbdata, typename todata>
        auto measure_gparam(const top& op, const tidata& idata, twdata& wdata, tbdata& bdata, const todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.gparam(idata, wdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_gparam() * count, duration);
        }

        bool benchmark(const int isize, const int osize, const int count, table_t& table)
        {
                const auto params = affine_params_t{isize, 1, 1, osize, 1, 1};
                const auto kflops_output = params.flops_output() / 1024;
                const auto kflops_ginput = params.flops_ginput() / 1024;
                const auto kflops_gparam = params.flops_gparam() / 1024;

                const auto config = strcat("isize=", isize, ",osize=", osize, ",count=", count);

                if (!params.valid())
                {
                        log_error() << "invalid parameters (" << config << ")!";
                        return false;
                }

                auto wdata = params.make_wdata(); wdata.setRandom();
                auto bdata = params.make_bdata(); bdata.setRandom();
                auto idata = params.make_idata(count); idata.setRandom();
                auto odata = params.make_odata(count); odata.setRandom();

                // 3D implementation
                const auto op3d = affine3d_t{params};
                const auto gf3d_output = measure_output(op3d, idata, wdata, bdata, odata);
                const auto gf3d_ginput = measure_ginput(op3d, idata, wdata, bdata, odata);
                const auto gf3d_gparam = measure_gparam(op3d, idata, wdata, bdata, odata);

                // 4D implementation
                const auto op4d = affine4d_t{params};
                const auto gf4d_output = measure_output(op4d, idata, wdata, bdata, odata);
                const auto gf4d_ginput = measure_ginput(op4d, idata, wdata, bdata, odata);
                const auto gf4d_gparam = measure_gparam(op4d, idata, wdata, bdata, odata);

                table.append()
                        << params.idims() << config << params.odims() << params.psize()
                        << kflops_output << kflops_ginput << kflops_gparam
                        << gf3d_output << gf3d_ginput << gf3d_gparam
                        << gf4d_output << gf4d_ginput << gf4d_gparam;
                return true;
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark affine operators");
        cmdline.add("", "min-isize",    "minimum input size [32, 1024]", "1024");
        cmdline.add("", "max-isize",    "maximum input size [32, 4096]", "1024");
        cmdline.add("", "min-osize",    "minimum output size [32, 1024]", "1024");
        cmdline.add("", "max-osize",    "maximum output size [32, 4096]", "1024");
        cmdline.add("", "min-count",    "minimum number of samples in minibatch [1, 16]",  "1");
        cmdline.add("", "max-count",    "maximum number of samples in minibatch [1, 128]", "128");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_min_isize = clamp(cmdline.get<int>("min-isize"), 32, 1024);
        const auto cmd_max_isize = clamp(cmdline.get<int>("max-isize"), cmd_min_isize, 4096);
        const auto cmd_min_osize = clamp(cmdline.get<int>("min-osize"), 32, 1024);
        const auto cmd_max_osize = clamp(cmdline.get<int>("max-osize"), cmd_min_osize, 4096);
        const auto cmd_min_count = clamp(cmdline.get<int>("min-count"), 1, 16);
        const auto cmd_max_count = clamp(cmdline.get<int>("max-count"), cmd_min_count, 128);

        table_t table;
        table.header()
                << colspan(4) << ""
                << colspan(3) << alignment::center << colfill('=') << "operations[#kflops]"
                << colspan(3) << alignment::center << colfill('=') << "3d kernel[gflop/s]"
                << colspan(3) << alignment::center << colfill('=') << "4d kernel[gflop/s]";
        table.delim();
        table.append()
                << "isize" << "config" << "osize" << "#params"
                << "output" << "ginput" << "gparam"
                << "output" << "ginput" << "gparam"
                << "output" << "ginput" << "gparam";
        table.delim();

        // benchmark for different input-output sizes and number of samples in a minibatch
        for (auto isize = cmd_min_isize; isize <= cmd_max_isize; isize *= 2)
        {
                for (auto osize = cmd_min_osize; osize <= cmd_max_osize; osize *= 2)
                {
                        for (auto count = cmd_min_count; count <= cmd_max_count; count *= 2)
                        {
                                benchmark(isize, osize, count, table);
                        }

                        if (osize * 2 <= cmd_max_osize)
                        {
                                table.delim();
                        }
                }

                if (isize * 2 <= cmd_max_isize)
                {
                        table.delim();
                }
        }

        // print results
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
