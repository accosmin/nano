#include "logger.h"
#include "measure.h"
#include "text/table.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "layers/conv3d_naive.h"
#include <iostream>

namespace nano
{
        using dim2d_t = tensor_dims_t<2>;

        template <typename tvalue>
        string_t serialize_to_string(const tvalue value)
        {
                std::stringstream s;
                s << value;
                return s.str();
        }

        template <> string_t to_string<dim2d_t>(const dim2d_t dims) { return serialize_to_string(dims); }
        template <> string_t to_string<dim3d_t>(const dim3d_t dims) { return serialize_to_string(dims); }
        template <> string_t to_string<dim4d_t>(const dim4d_t dims) { return serialize_to_string(dims); }
}

namespace
{
        const size_t trials = 16;

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_output(const top& op, const tidata& idata, const tkdata& kdata, const tbdata& bdata, todata& odata)
        {
                return nano::measure_robustly_usec([&] () { op.output(idata, kdata, bdata, odata); }, trials);
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_ginput(const top& op, tidata& idata, const tkdata& kdata, const tbdata& bdata, const todata& odata)
        {
                return nano::measure_robustly_usec([&] () { op.ginput(idata, kdata, bdata, odata); }, trials);
        }

        template <typename top, typename tidata, typename tkdata, typename tbdata, typename todata>
        auto measure_gparam(const top& op, const tidata& idata, tkdata& kdata, tbdata& bdata, const todata& odata)
        {
                return nano::measure_robustly_usec([&] () { op.gparam(idata, kdata, bdata, odata); }, trials);
        }
}

int main(int argc, const char *argv[])
{
        using namespace nano;

        // parse the command line
        cmdline_t cmdline("benchmark models");
        cmdline.add("", "imaps",        "number of input planes [1,128]", "32");
        cmdline.add("", "irows",        "number of input rows [16, 128]", "24");
        cmdline.add("", "icols",        "number of input cols [16, 128]", "24");
        cmdline.add("", "omaps",        "number of output planes [1, 128]", "32");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_imaps = clamp(cmdline.get<int>("imaps"), 1, 128);
        const auto cmd_irows = clamp(cmdline.get<int>("irows"), 16, 128);
        const auto cmd_icols = clamp(cmdline.get<int>("icols"), 16, 128);
        const auto cmd_omaps = clamp(cmdline.get<int>("omaps"), 1, 128);

        // benchmark 3D convolutions various kernel sizes & connectivity factors
        table_t table;
        table.header()
                << "isize" << "conn" << "kernel" << "stride" << "osize" << "params" << "kflops"
                << "output[us]" << "ginput[us]" << "gparam[us]";

        for (tensor_size_t ksize = 3; ksize <= 9; ksize += 2)
        {
                for (tensor_size_t kdelta = 1; kdelta <= 2; ++ kdelta)
                {
                        for (tensor_size_t kconn = 1; kconn <= 8; kconn *= 2)
                        {
                                const auto params = conv3d_params_t
                                {
                                        cmd_imaps, cmd_irows, cmd_icols,
                                        cmd_omaps, kconn, ksize, ksize, kdelta, kdelta
                                };

                                auto bdata = params.make_bdata(); bdata.setRandom();
                                auto idata = params.make_idata(); idata.vector().setRandom();
                                auto kdata = params.make_kdata(); kdata.vector().setRandom();
                                auto odata = params.make_odata(); odata.vector().setRandom();

                                const auto op_naive = conv3d_naive_t{params};
                                const auto us_naive_output = measure_output(op_naive, idata, kdata, bdata, odata);
                                const auto us_naive_ginput = measure_ginput(op_naive, idata, kdata, bdata, odata);
                                const auto us_naive_gparam = measure_gparam(op_naive, idata, kdata, bdata, odata);

                                table.append()
                                        << dim3d_t{params.imaps(), params.irows(), params.icols()}
                                        << params.kconn()
                                        << dim2d_t{params.krows(), params.kcols()}
                                        << dim2d_t{params.kdrow(), params.kdcol()}
                                        << dim3d_t{params.omaps(), params.orows(), params.ocols()}
                                        << params.psize()
                                        << (params.flops() / 1024)
                                        << us_naive_output.count()
                                        << us_naive_ginput.count()
                                        << us_naive_gparam.count();
                        }
                }
        }

        // print results
        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
