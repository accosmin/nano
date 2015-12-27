#include "text/table.h"
#include "cortex/tensor.h"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "tensor/conv2d_cpp.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/conv2d_eig.hpp"
#include "tensor/corr2d_cpp.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "tensor/corr2d_egb.hpp"
#include "tensor/corr2d_egr.hpp"
#include "text/table_row_mark.h"
#include "cortex/util/measure.hpp"
#include <iostream>
#include <boost/program_options.hpp>

namespace
{
        using namespace cortex;

        std::string make_header(const int isize, const int ksize)
        {
                const int osize = isize - ksize + 1;

                return  "(" +
                        text::to_string(isize) + "x" +
                        text::to_string(isize) + " @ " +
                        text::to_string(ksize) + "x" +
                        text::to_string(ksize) + " -> " +
                        text::to_string(osize) + "x" +
                        text::to_string(osize) +
                        ")";
        }

        template
        <
                typename tmatrix
        >
        void make_matrices(const int isize, const int ksize,
                tmatrix& idata, tmatrix& kdata, tmatrix& odata)
        {
                const int osize = isize - ksize + 1;

                math::random_t<typename tmatrix::Scalar> rng(-1.0 / isize, 1.0 / isize);

                idata.resize(isize, isize);
                kdata.resize(ksize, ksize);
                odata.resize(osize, osize);

                tensor::set_random(idata, rng);
                tensor::set_random(kdata, rng);
                tensor::set_random(odata, rng);
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
                return cortex::measure_robustly_usec([&] ()
                {
                        op(idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename tmatrix
        >
        void test_config_conv(const int isize, const int ksize, text::table_row_t& row, const size_t trials = 16)
        {
                tmatrix idata, kdata, odata;
                make_matrices(isize, ksize, idata, kdata, odata);

                tmatrix odata_ret = odata;

                row << measure_op(tensor::conv2d_eig_t(), idata, kdata, odata_ret, trials).count();
                row << measure_op(tensor::conv2d_cpp_t(), idata, kdata, odata_ret, trials).count();
                row << measure_op(tensor::conv2d_dot_t(), idata, kdata, odata_ret, trials).count();
                row << measure_op(tensor::conv2d_mad_t(), idata, kdata, odata_ret, trials).count();
                row << measure_op(tensor::conv2d_dyn_t(), idata, kdata, odata_ret, trials).count();
        }

        template
        <
                typename tmatrix
        >
        void test_config_corr(const int isize, const int ksize, text::table_row_t& row, const size_t trials = 16)
        {
                tmatrix idata, kdata, odata;
                make_matrices(isize, ksize, idata, kdata, odata);

                tmatrix idata_ret = idata;

                row << measure_op(tensor::corr2d_egb_t(), odata, kdata, idata_ret, trials).count();
                row << measure_op(tensor::corr2d_egr_t(), odata, kdata, idata_ret, trials).count();
                row << measure_op(tensor::corr2d_cpp_t(), odata, kdata, idata_ret, trials).count();
                row << measure_op(tensor::corr2d_mdk_t(), odata, kdata, idata_ret, trials).count();
                row << measure_op(tensor::corr2d_mdo_t(), odata, kdata, idata_ret, trials).count();
                row << measure_op(tensor::corr2d_dyn_t(), odata, kdata, idata_ret, trials).count();
        }
}

int main(int argc, char* argv[])
{
        using namespace cortex;

        const int min_isize = 4;
        const int max_isize = 48;

        const int min_ksize = 3;
        const int max_ksize = 15;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h", "benchmark 2D convolutions & correlations");
        po_desc.add_options()("conv", "benchmark convolutions");
        po_desc.add_options()("corr", "benchmark correlations");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const bool has_conv = po_vm.count("conv") > 0;
        const bool has_corr = po_vm.count("corr") > 0;

        // convolutions
        if (has_conv)
        {
                text::table_t table("size\\convolution [us]");
                table.header()
                        << "2D (eig)"
                        << "2D (cpp)"
                        << "2D (dot)"
                        << "2D (mad)"
                        << "2D (dyn)";

                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const auto header = make_header(isize, ksize);
                                test_config_conv<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(text::make_table_mark_minimum_col<size_t>());
                table.print(std::cout);
        }

        // correlations
        if (has_corr)
        {
                text::table_t table("size\\correlation [us]");
                table.header()
                        << "2D (egb)"
                        << "2D (egr)"
                        << "2D (cpp)"
                        << "2D (mkd)"
                        << "2D (mko)"
                        << "2D (dyn)";

                for (int isize = min_isize; isize <= max_isize; isize += 4)
                {
                        for (int ksize = min_ksize; ksize <= std::min(isize, max_ksize); ksize += 2)
                        {
                                const auto header = make_header(isize, ksize);
                                test_config_corr<matrix_t>(isize, ksize, table.append(header));
                        }
                }

                table.mark(text::make_table_mark_minimum_col<size_t>());
                table.print(std::cout);
        }

	return EXIT_SUCCESS;
}

