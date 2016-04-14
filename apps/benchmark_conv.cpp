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

namespace
{
        using namespace nano;

        template
        <
                typename tmatrix
        >
        void make_matrices(const int isize, const int ksize,
                tmatrix& idata, tmatrix& kdata, tmatrix& odata)
        {
                const int osize = isize - ksize + 1;

                nano::random_t<typename tmatrix::Scalar> rng(-1.0 / isize, 1.0 / isize);

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
        auto measure_op(const top& op,
                const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata, const size_t trials = 16)
        {
                return (nano::measure_robustly_nsec([&] ()
                {
                        op(idata, kdata, odata);
                }, trials).count() * 1000 + kdata.size() * odata.size() - 1) / (kdata.size() * odata.size());
        }

        template
        <
                typename tmatrix
        >
        void test_config_conv(const int isize, const int ksize, nano::table_row_t& row, const size_t trials = 16)
        {
                tmatrix idata, kdata, odata;
                make_matrices(isize, ksize, idata, kdata, odata);

                row << measure_op(tensor::conv2d_eig_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_cpp_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_dot_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_dot_dyn_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_mad_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_mad_dyn_t(), idata, kdata, odata, trials);
                row << measure_op(tensor::conv2d_dyn_t(), idata, kdata, odata, trials);
        }

        template
        <
                typename tmatrix
        >
        void test_config_corr(const int isize, const int ksize, nano::table_row_t& row, const size_t trials = 16)
        {
                tmatrix idata, kdata, odata;
                make_matrices(isize, ksize, idata, kdata, odata);

                row << measure_op(tensor::corr2d_egb_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_egr_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_cpp_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_mdk_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_mdk_dyn_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_mdo_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_mdo_dyn_t(), odata, kdata, idata, trials);
                row << measure_op(tensor::corr2d_dyn_t(), odata, kdata, idata, trials);
        }
}

int main(int argc, const char* argv[])
{
        using namespace nano;

        const int min_ksize = 3;
        const int max_ksize = 15;

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
        const int min_isize = cmdline.get<int>("min-isize");
        const int max_isize = cmdline.get<int>("max-isize");

        if (!has_conv && !has_corr)
        {
                cmdline.usage();
        }

        // convolutions
        if (has_conv)
        {
                nano::table_t table("size\\method [ps/mul]");
                table.header()
                        << "eig"
                        << "cpp"
                        << "dot"
                        << "dot-dyn"
                        << "mad"
                        << "mad-dyn"
                        << "dyn";

                for (int isize = min_isize; isize <= max_isize; ++ isize)
                {
                        std::set<int> ksizes;
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                ksizes.insert(ksize);
                                ksizes.insert(isize - ksize + 1);
                        }

                        for (auto ksize : ksizes)
                        {
                                const int osize = isize - ksize + 1;
                                const auto header =
                                        "(" + nano::to_string(isize) + "x" + nano::to_string(isize) + " @ " +
                                              nano::to_string(ksize) + "x" + nano::to_string(ksize) + " -> " +
                                              nano::to_string(osize) + "x" + nano::to_string(osize) + ")";

                                test_config_conv<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(nano::make_table_mark_minimum_percentage_cols<size_t>(10));
                table.print(std::cout);
        }

        // correlations
        if (has_corr)
        {
                nano::table_t table("size\\method [ps/mul]");
                table.header()
                        << "egb"
                        << "egr"
                        << "cpp"
                        << "mdk"
                        << "mdk-dyn"
                        << "mdo"
                        << "mdo-dyn"
                        << "dyn";

                for (int isize = min_isize; isize <= max_isize; ++ isize)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const int osize = isize - ksize + 1;
                                const auto header =
                                        "(" + nano::to_string(osize) + "x" + nano::to_string(osize) + " @ " +
                                              nano::to_string(ksize) + "x" + nano::to_string(ksize) + " -> " +
                                              nano::to_string(isize) + "x" + nano::to_string(isize) + ")";

                                test_config_corr<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(nano::make_table_mark_minimum_percentage_cols<size_t>(10));
                table.print(std::cout);
        }

        return EXIT_SUCCESS;
}

