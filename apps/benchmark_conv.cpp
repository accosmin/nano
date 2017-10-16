#include "logger.h"
#include "text/table.h"
#include "text/config.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "chrono/measure.h"
#include "layers/conv3d.h"
#include "layers/conv4d.h"
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

namespace
{
        const auto trials = size_t(16);

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_output(top& op, const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.output(idata, kdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_output() * count, duration);
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_ginput(top& op, tidata& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.ginput(idata, kdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_ginput() * count, duration);
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_gparam(top& op, const tidata& idata, tkdata& kdata, tbdata& bdata, const todata& odata)
        {
                const auto count = idata.template size<0>();
                const auto duration = measure<nanoseconds_t>([&] () { op.gparam(idata, kdata, bdata, odata); }, trials);
                return nano::gflops(op.params().flops_gparam() * count, duration);
        }

        bool benchmark(const int imaps, const int irows, const int icols, const int omaps,
                const int ksize, const int kdelta, const int kconn, const int count, table_t& table)
        {
                const auto params = conv_params_t
                {
                        imaps, irows, icols,
                        omaps, kconn, ksize, ksize, kdelta, kdelta
                };

                const auto kflops_output = params.flops_output() / 1024;
                const auto kflops_ginput = params.flops_ginput() / 1024;
                const auto kflops_gparam = params.flops_gparam() / 1024;

                const auto config = to_params(
                        "conn", kconn, "rows", ksize, "cols", ksize, "drow", kdelta, "dcol", kdelta, "count", count);

                if (!params.valid())
                {
                        log_error() << "invalid parameters (" << config << ")!";
                        return false;
                }

                auto bdata = params.make_bdata(); bdata.setRandom();
                auto kdata = params.make_kdata(); kdata.setRandom();
                auto idata = params.make_idata(count); idata.setRandom();
                auto odata = params.make_odata(count); odata.setRandom();

                // 3D implementation
                const auto op3d = conv3d_t{params};
                const auto gf3d_output = measure_output(op3d, idata, kdata, bdata, odata);
                const auto gf3d_ginput = measure_ginput(op3d, idata, kdata, bdata, odata);
                const auto gf3d_gparam = measure_gparam(op3d, idata, kdata, bdata, odata);

                // 4D implementation
                auto op4d = conv4d_t{params};
                op4d.output(idata, kdata, bdata, odata);// NB: needed to update the internal buffers!
                const auto gf4d_output = measure_output(op4d, idata, kdata, bdata, odata);
                const auto gf4d_ginput = measure_ginput(op4d, idata, kdata, bdata, odata);
                const auto gf4d_gparam = measure_gparam(op4d, idata, kdata, bdata, odata);

                table.append()
                        << tensor3d_dims_t{params.imaps(), params.irows(), params.icols()}
                        << config
                        << tensor3d_dims_t{params.omaps(), params.orows(), params.ocols()}
                        << params.psize()
                        << kflops_output << kflops_ginput << kflops_gparam
                        << gf3d_output << gf3d_ginput << gf3d_gparam
                        << gf4d_output << gf4d_ginput << gf4d_gparam;

                return true;
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark 3D convolution operators");
        cmdline.add("", "imaps",        "number of input planes [1,128]", "32");
        cmdline.add("", "irows",        "number of input rows [16, 128]", "24");
        cmdline.add("", "icols",        "number of input cols [16, 128]", "24");
        cmdline.add("", "omaps",        "number of output planes [1, 128]", "32");
        cmdline.add("", "min-kconn",    "minimum connectivity factor [1, 16]", "1");
        cmdline.add("", "max-kconn",    "maximum connectivity factor [1, 16]", "4");
        cmdline.add("", "min-ksize",    "minimum kernel size [1, 15]", "1");
        cmdline.add("", "max-ksize",    "maximum kernel size [1, 15]", "9");
        cmdline.add("", "min-kdelta",   "minimum kernel stride [1, 3]", "1");
        cmdline.add("", "max-kdelta",   "maximum kernel stride [1, 3]", "2");
        cmdline.add("", "min-count",    "minimum number of samples in minibatch [1, 16]",  "1");
        cmdline.add("", "max-count",    "maximum number of samples in minibatch [1, 128]", "128");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_imaps = clamp(cmdline.get<int>("imaps"), 1, 128);
        const auto cmd_irows = clamp(cmdline.get<int>("irows"), 16, 128);
        const auto cmd_icols = clamp(cmdline.get<int>("icols"), 16, 128);
        const auto cmd_omaps = clamp(cmdline.get<int>("omaps"), 1, 128);
        const auto cmd_min_kconn = clamp(cmdline.get<int>("min-kconn"), 1, 16);
        const auto cmd_max_kconn = clamp(cmdline.get<int>("max-kconn"), cmd_min_kconn, 16);
        const auto cmd_min_ksize = clamp(cmdline.get<int>("min-ksize"), 1, 15);
        const auto cmd_max_ksize = clamp(cmdline.get<int>("max-ksize"), cmd_min_ksize, 15);
        const auto cmd_min_kdelta = clamp(cmdline.get<int>("min-kdelta"), 1, 3);
        const auto cmd_max_kdelta = clamp(cmdline.get<int>("max-kdelta"), cmd_min_kdelta, 3);
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

        // benchmark for different kernel sizes, connectivity factors and number of samples in a minibatch
        for (auto ksize = cmd_min_ksize; ksize <= cmd_max_ksize; ksize += 2)
        {
                for (auto kdelta = cmd_min_kdelta; kdelta <= cmd_max_kdelta; ++ kdelta)
                {
                        for (auto kconn = cmd_min_kconn; kconn <= cmd_max_kconn; kconn *= 2)
                        {
                                for (auto count = cmd_min_count; count <= cmd_max_count; count *= 2)
                                {
                                        if (!benchmark(cmd_imaps, cmd_irows, cmd_icols, cmd_omaps,
                                                       ksize, kdelta, kconn, count, table))
                                        {
                                                break;
                                        }
                                }

                                if (kconn + 1 <= cmd_max_kconn)
                                {
                                        table.delim();
                                }
                        }

                        if (kdelta + 1 <= cmd_max_kdelta)
                        {
                                table.delim();
                        }
                }

                if (ksize + 2 <= cmd_max_ksize)
                {
                        table.delim();
                }
        }

        // print results
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
