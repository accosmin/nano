#include "logger.h"
#include "text/table.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "layers/conv3d_params.h"
#include <iostream>

namespace nano
{
        template <> string_t to_string<dim3d_t>(const dim3d_t dims)
        {
                std::stringstream s;
                s << dims;
                return s.str();
        }

        template <> string_t to_string<dim4d_t>(const dim4d_t dims)
        {
                std::stringstream s;
                s << dims;
                return s.str();
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
        table.header() << "isize" << "kconn" << "ksize" << "osize" << "params" << "kflops";

        for (tensor_size_t ksize = 3; ksize <= 9; ksize += 2)
        {
                for (tensor_size_t kdelta = 1; kdelta <= 2; ++ kdelta)
                {
                        for (tensor_size_t kconn = 1; kconn <= 8; kconn *= 2)
                        {
                                const auto param = conv3d_params_t{
                                        cmd_imaps, cmd_irows, cmd_icols,
                                        cmd_omaps, kconn,
                                        ksize, ksize, kdelta, kdelta};

                                table.append()
                                        << dim3d_t{param.imaps(), param.irows(), param.icols()}
                                        << param.kconn()
                                        << dim4d_t{param.krows(), param.kcols(), param.kdrow(), param.kdcol()}
                                        << dim3d_t{param.omaps(), param.orows(), param.ocols()}
                                        << param.psize()
                                        << (param.flops() / 1024);
                        }
                }
        }

        // print results
        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
