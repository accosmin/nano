#include "tensor.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/random.hpp"
#include "tensor/numeric.hpp"
#include "cortex/measure.hpp"
#include "tensor/conv2d_cpp.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/conv2d_eig.hpp"
#include "tensor/corr2d_cpp.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "tensor/corr2d_egb.hpp"
#include "tensor/corr2d_egr.hpp"
#include "text/table_row_mark.h"
#include <set>
#include <iostream>

using namespace nano;

template
<
        typename tmatrix
>
void make_matrices(const tensor_size_t isize, const tensor_size_t ksize,
        tmatrix& idata, tmatrix& kdata, tmatrix& odata)
{
        const tensor_size_t osize = isize - ksize + 1;

        nano::random_t<scalar_t> rng(-scalar_t(1) / scalar_t(isize), +scalar_t(1) / scalar_t(isize));

        idata.resize(isize, isize);
        kdata.resize(ksize, ksize);
        odata.resize(osize, osize);

        tensor::set_random(rng, idata, kdata, odata);
}

template
<
        typename top,
        typename tmatrixi,
        typename tmatrixk,
        typename tmatrixo
>
auto measure_op(const tensor_size_t flops, const top& op,
        const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata, const size_t trials = 16)
{
        const auto duration = nano::measure_robustly_nsec([&] ()
        {
                op(idata, kdata, odata);
        }, trials);

        return nano::gflops(flops, duration);
}

template
<
        typename tmatrix
>
void eval_conv(const tensor_size_t isize, const tensor_size_t ksize, nano::table_row_t& row, const size_t trials = 16)
{
        tmatrix idata, kdata, odata;
        make_matrices(isize, ksize, idata, kdata, odata);

        const auto flops = kdata.size() * odata.size();

        row << measure_op(flops, tensor::conv2d_eig_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_cpp_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_dot_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_dot_dyn_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_mad_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_mad_dyn_t(), idata, kdata, odata, trials);
        row << measure_op(flops, tensor::conv2d_dyn_t(), idata, kdata, odata, trials);
}

template
<
        typename tmatrix
>
void eval_corr(const tensor_size_t isize, const tensor_size_t ksize, nano::table_row_t& row, const size_t trials = 16)
{
        tmatrix idata, kdata, odata;
        make_matrices(isize, ksize, idata, kdata, odata);

        const auto flops = kdata.size() * odata.size();

        row << measure_op(flops, tensor::corr2d_egb_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_egr_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_cpp_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_mdk_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_mdk_dyn_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_mdo_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_mdo_dyn_t(), odata, kdata, idata, trials);
        row << measure_op(flops, tensor::corr2d_dyn_t(), odata, kdata, idata, trials);
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        const tensor_size_t min_ksize = 3;
        const tensor_size_t max_ksize = 15;

        // parse the command line
        nano::cmdline_t cmdline("benchmark 2D convolutions & correlations");
        cmdline.add("", "conv",         "benchmark convolutions");
        cmdline.add("", "corr",         "benchmark correlations");
        cmdline.add("", "min-isize",    "minimum input size (pixels)", "4");
        cmdline.add("", "max-isize",    "maximum input size (pixels)", "48");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto has_conv = cmdline.has("conv");
        const auto has_corr = cmdline.has("corr");
        const tensor_size_t min_isize = cmdline.get<tensor_size_t>("min-isize");
        const tensor_size_t max_isize = cmdline.get<tensor_size_t>("max-isize");

        if (!has_conv && !has_corr)
        {
                cmdline.usage();
        }

        // convolutions
        if (has_conv)
        {
                nano::table_t table("size\\method [GFLOPS]");
                table.header()
                        << "eig"
                        << "cpp"
                        << "dot"
                        << "dot-dyn"
                        << "mad"
                        << "mad-dyn"
                        << "dyn";

                for (tensor_size_t isize = min_isize; isize <= max_isize; ++ isize)
                {
                        std::set<tensor_size_t> ksizes;
                        for (tensor_size_t ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                ksizes.insert(ksize);
                                ksizes.insert(isize - ksize + 1);
                        }

                        for (auto ksize : ksizes)
                        {
                                const tensor_size_t osize = isize - ksize + 1;
                                const auto header =
                                        "(" + nano::to_string(isize) + "x" + nano::to_string(isize) + " @ " +
                                              nano::to_string(ksize) + "x" + nano::to_string(ksize) + " -> " +
                                              nano::to_string(osize) + "x" + nano::to_string(osize) + ")";

                                eval_conv<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(nano::make_table_mark_maximum_percentage_cols<size_t>(10));
                table.print(std::cout);
        }

        // correlations
        if (has_corr)
        {
                nano::table_t table("size\\method [GFLOPS]");
                table.header()
                        << "egb"
                        << "egr"
                        << "cpp"
                        << "mdk"
                        << "mdk-dyn"
                        << "mdo"
                        << "mdo-dyn"
                        << "dyn";

                for (tensor_size_t isize = min_isize; isize <= max_isize; ++ isize)
                {
                        for (tensor_size_t ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const tensor_size_t osize = isize - ksize + 1;
                                const auto header =
                                        "(" + nano::to_string(osize) + "x" + nano::to_string(osize) + " @ " +
                                              nano::to_string(ksize) + "x" + nano::to_string(ksize) + " -> " +
                                              nano::to_string(isize) + "x" + nano::to_string(isize) + ")";

                                eval_corr<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(nano::make_table_mark_maximum_percentage_cols<size_t>(10));
                table.print(std::cout);
        }

        return EXIT_SUCCESS;
}

