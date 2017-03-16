#include "logger.h"
#include "measure.h"
#include "text/table.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "text/to_params.h"
#include "layers/conv3d_naive.h"
#include "layers/conv3d_toeplitz.h"
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
        template <typename top>
        auto measure(const top& op, const tensor_size_t flops)
        {
                const auto trials = size_t(16);
                const auto duration = measure_robustly_usec([&] () { op(); }, trials);
                return nano::gflops(flops, duration);
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_output(const top& op, const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata& odata)
        {
                return ::measure([&] () { op.output(idata, kdata, bdata, odata); }, op.params().flops_output());
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_ginput(const top& op, tidata& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata)
        {
                return ::measure([&] () { op.ginput(idata, kdata, bdata, odata); }, op.params().flops_ginput());
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_gparam(const top& op, const tidata& idata, tkdata& kdata, tbdata& bdata, const todata& odata)
        {
                return ::measure([&] () { op.gparam(idata, kdata, bdata, odata); }, op.params().flops_gparam());
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark models");
        cmdline.add("", "imaps",        "number of input planes [1,128]", "32");
        cmdline.add("", "irows",        "number of input rows [16, 128]", "24");
        cmdline.add("", "icols",        "number of input cols [16, 128]", "24");
        cmdline.add("", "omaps",        "number of output planes [1, 128]", "32");
        cmdline.add("", "min-kconn",    "minimum connectivity factor [1, 16]", "1");
        cmdline.add("", "max-kconn",    "maximum connectivity factor [1, 16]", "4");
        cmdline.add("", "min-ksize",    "minimum kernel size [3, 15]", "3");
        cmdline.add("", "max-ksize",    "maximum kernel size [3, 15]", "9");

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
        const auto cmd_min_kdelta = 1;
        const auto cmd_max_kdelta = 2;

        table_t table;
        table.header()
                << "isize" << "convolution parameters" << "osize" << "params"
                << "output[kflops]" << "ginput" << "gparam"
                << "naive[gflop/s]" << "ginput" << "gparam"
                << "toepl[gflop/s]" << "ginput" << "gparam";

        // benchmark 3D convolutions various kernel sizes & connectivity factors
        for (tensor_size_t ksize = cmd_min_ksize; ksize <= cmd_max_ksize; ksize += 2)
        {
                for (tensor_size_t kdelta = cmd_min_kdelta; kdelta <= cmd_max_kdelta; ++ kdelta)
                {
                        for (tensor_size_t kconn = cmd_min_kconn; kconn <= cmd_max_kconn; kconn *= 2)
                        {
                                const auto params = conv3d_params_t
                                {
                                        cmd_imaps, cmd_irows, cmd_icols,
                                        cmd_omaps, kconn, ksize, ksize, kdelta, kdelta
                                };

                                const auto kflops_output = params.flops_output() / 1024;
                                const auto kflops_ginput = params.flops_ginput() / 1024;
                                const auto kflops_gparam = params.flops_gparam() / 1024;

                                const auto config = to_params("conn", kconn,
                                        "rows", ksize, "cols", ksize, "drow", kdelta, "dcol", kdelta);

                                if (!params.valid())
                                {
                                        log_error() << "invalid parameters (" << config << ")!";
                                        break;
                                }

                                auto bdata = params.make_bdata(); bdata.setRandom();
                                auto idata = params.make_idata(); idata.vector().setRandom();
                                auto kdata = params.make_kdata(); kdata.vector().setRandom();
                                auto odata = params.make_odata(); odata.vector().setRandom();

                                // naive implementation
                                const auto op_naive = conv3d_naive_t{params};
                                const auto gf_naive_output = measure_output(op_naive, idata, kdata, bdata, odata);
                                const auto gf_naive_ginput = measure_ginput(op_naive, idata, kdata, bdata, odata);
                                const auto gf_naive_gparam = measure_gparam(op_naive, idata, kdata, bdata, odata);

                                // Toeplitz implementation
                                const auto op_toepl = conv3d_toeplitz_t{params};
                                const auto gf_toepl_output = measure_output(op_toepl, idata, kdata, bdata, odata);
                                const auto gf_toepl_ginput = measure_ginput(op_toepl, idata, kdata, bdata, odata);
                                const auto gf_toepl_gparam = measure_gparam(op_toepl, idata, kdata, bdata, odata);

                                table.append()
                                        << tensor3d_dims_t{params.imaps(), params.irows(), params.icols()}
                                        << config
                                        << tensor3d_dims_t{params.omaps(), params.orows(), params.ocols()}
                                        << params.psize()
                                        << kflops_output << kflops_ginput << kflops_gparam
                                        << gf_naive_output << gf_naive_ginput << gf_naive_gparam
                                        << gf_toepl_output << gf_toepl_ginput << gf_toepl_gparam;
                        }

                        if (kdelta + 1 <= cmd_max_kdelta)
                        {
                                table.append(table_row_t::storage::delim);
                        }
                }

                if (ksize + 2 <= cmd_max_ksize)
                {
                        table.append(table_row_t::storage::delim);
                }
        }

        // print results
        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
