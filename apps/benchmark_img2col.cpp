#include "logger.h"
#include "text/table.h"
#include "text/config.h"
#include "math/numeric.h"
#include "text/cmdline.h"
#include "chrono/measure.h"
#include "layers/conv_utils.h"
#include "layers/conv_params.h"
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

        template <typename top>
        auto measure_img2col(const top& op,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols)
        {
                const auto flops = orows * ocols * krows * kcols;
                const auto duration = measure<picoseconds_t>([&] () { op(); }, trials);
                return nano::gflops(flops, duration);
        }

        template <typename timatrix, typename tomatrix>
        auto measure_img2col0(const timatrix& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                tomatrix&& omat)
        {
                return  measure_img2col([&] () { nano::img2col0(imat, orows, ocols, krows, kcols, drows, dcols, omat); },
                        orows, ocols, krows, kcols);
        }

        template <typename timatrix, typename tomatrix>
        auto measure_img2colx(const timatrix& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                tomatrix&& omat)
        {
                return  measure_img2col([&] () { nano::img2colx(imat, orows, ocols, krows, kcols, drows, dcols, omat); },
                        orows, ocols, krows, kcols);
        }

        template <typename top>
        auto measure_col2img(const top& op,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols)
        {
                const auto flops = 2 * orows * ocols * krows * kcols;
                const auto duration = measure<picoseconds_t>([&] () { op(); }, trials);
                return nano::gflops(flops, duration);
        }

        template <typename timatrix, typename tomatrix>
        auto measure_col2img(timatrix&& imat,
                const tensor_size_t orows, const tensor_size_t ocols,
                const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t drows, const tensor_size_t dcols,
                const tomatrix& omat)
        {
                return  measure_col2img([&] { nano::col2img(imat, orows, ocols, krows, kcols, drows, dcols, omat); },
                        orows, ocols, krows, kcols);
        }

        bool benchmark(const int imaps, const int irows, const int icols, const int omaps,
                const int ksize, const int kdelta, const int kconn, table_t& table)
        {
                const auto params = conv_params_t
                {
                        imaps, irows, icols,
                        omaps, kconn, ksize, ksize, kdelta, kdelta
                };

                const auto orows = params.orows(), ocols = params.ocols();
                const auto krows = params.krows(), kcols = params.kcols();
                const auto drows = params.kdrow(), dcols = params.kdcol();

                const auto config = to_params(
                        "conn", kconn, "rows", ksize, "cols", ksize, "drow", kdelta, "dcol", kdelta);

                if (!params.valid())
                {
                        log_error() << "invalid parameters (" << config << ")!";
                        return false;
                }

                // img2col implementation
                matrix_t imap(irows, icols); imap.setRandom();
                matrix_t omap(krows * kcols, orows * ocols); omap.setRandom();

                const auto img2col0 = measure_img2col0(imap, orows, ocols, krows, kcols, drows, dcols, omap);
                const auto img2colx = measure_img2colx(imap, orows, ocols, krows, kcols, drows, dcols, omap);

                const auto col2img = measure_col2img(imap, orows, ocols, krows, kcols, drows, dcols, omap);

                table.append()
                        << tensor3d_dims_t{params.imaps(), params.irows(), params.icols()}
                        << config
                        << tensor3d_dims_t{params.omaps(), params.orows(), params.ocols()}
                        << img2col0 << img2colx << col2img;

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

        table_t table;
        table.header()
                << "" << "" << ""
                << "img2col" << "img2colx" << "col2img";

        table.append()
                << "isize" << "config" << "osize"
                << "gflop/s" << "gflop/s" << "gflop/s";

        table.append(table_row_t::storage::delim);

        // benchmark for different kernel sizes, connectivity factors and number of samples in a minibatch
        for (auto ksize = cmd_min_ksize; ksize <= cmd_max_ksize; ksize += 2)
        {
                for (auto kdelta = cmd_min_kdelta; kdelta <= cmd_max_kdelta; ++ kdelta)
                {
                        for (auto kconn = cmd_min_kconn; kconn <= cmd_max_kconn; kconn *= 2)
                        {
                                if (!benchmark(cmd_imaps, cmd_irows, cmd_icols, cmd_omaps,
                                               ksize, kdelta, kconn, table))
                                {
                                        break;
                                }
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
        return EXIT_SUCCESS;
}
