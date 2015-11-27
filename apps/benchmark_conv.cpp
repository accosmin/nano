#include "text/table.h"
#include "cortex/tensor.h"
#include "math/random.hpp"
#include "tensor/conv3d.hpp"
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

        std::string make_header(const int idims, const int isize, const int ksize, const int odims)
        {
                const int osize = isize - ksize + 1;

                return  "(" +
                        text::to_string(idims) + "x" +
                        text::to_string(isize) + "x" +
                        text::to_string(isize) + " @ " +
                        text::to_string(ksize) + "x" +
                        text::to_string(ksize) + " -> " +
                        text::to_string(odims) + "x" +
                        text::to_string(osize) + "x" +
                        text::to_string(osize) +
                        ")";
        }

        template
        <
                typename ttensor
        >
        void make_tensors(const int isize, const int idims, const int ksize, const int odims,
                ttensor& idata, ttensor& kdata, ttensor& odata)
        {
                const int osize = isize - ksize + 1;
                const int kdims = odims * idims;

                math::random_t<typename ttensor::Scalar> rng(-1.0 / isize, 1.0 / isize);

                idata.resize(idims, isize, isize);
                kdata.resize(kdims, ksize, ksize);
                odata.resize(odims, osize, osize);

                tensor::set_random(idata, rng);
                tensor::set_random(kdata, rng);
                tensor::set_random(odata, rng);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_output(const top& op,
                const ttensori& idata, const ttensork& kdata, ttensoro&& odata, const size_t trials = 16)
        {
                return cortex::measure_robustly_usec([&] ()
                {
                        tensor::conv3d_output(op, idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_ginput(const top& op,
                ttensori&& idata, const ttensork& kdata, const ttensoro& odata, const size_t trials = 16)
        {
                return cortex::measure_robustly_usec([&] ()
                {
                        tensor::conv3d_ginput(op, idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename top,
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        size_t measure_gparam(const top& op,
                const ttensori& idata, ttensork&& kdata, const ttensoro& odata, const size_t trials = 16)
        {
                return cortex::measure_robustly_usec([&] ()
                {
                        tensor::conv3d_gparam(op, idata, kdata, odata);
                }, trials);
        }

        template
        <
                typename ttensor
        >
        void test_config_output(const int isize, const int idims, const int ksize, const int odims,
                text::table_row_t& row, const size_t trials = 16)
        {
                ttensor idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                ttensor odata_ret = odata;

                row << measure_output(tensor::conv2d_eig_t(), idata, kdata, odata_ret, trials);
                row << measure_output(tensor::conv2d_cpp_t(), idata, kdata, odata_ret, trials);
                row << measure_output(tensor::conv2d_dot_t(), idata, kdata, odata_ret, trials);
                row << measure_output(tensor::conv2d_mad_t(), idata, kdata, odata_ret, trials);
                row << measure_output(tensor::conv2d_dyn_t(), idata, kdata, odata_ret, trials);
        }

        template
        <
                typename ttensor
        >
        void test_config_ginput(const int isize, const int idims, const int ksize, const int odims,
                text::table_row_t& row, const size_t trials = 16)
        {
                ttensor idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                ttensor idata_ret = idata;

                row << measure_ginput(tensor::corr2d_egb_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(tensor::corr2d_egr_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(tensor::corr2d_cpp_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(tensor::corr2d_mdk_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(tensor::corr2d_mdo_t(), idata_ret, kdata, odata, trials);
                row << measure_ginput(tensor::corr2d_dyn_t(), idata_ret, kdata, odata, trials);
        }

        template
        <
                typename ttensor
        >
        void test_config_gparam(const int isize, const int idims, const int ksize, const int odims,
                text::table_row_t& row, const size_t trials = 16)
        {
                ttensor idata, kdata, odata;
                make_tensors(isize, idims, ksize, odims, idata, kdata, odata);

                ttensor kdata_ret = kdata;

                row << measure_gparam(tensor::conv2d_eig_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(tensor::conv2d_cpp_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(tensor::conv2d_dot_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(tensor::conv2d_mad_t(), idata, kdata_ret, odata, trials);
                row << measure_gparam(tensor::conv2d_dyn_t(), idata, kdata_ret, odata, trials);
        }
}

int main(int argc, char* argv[])
{
        using namespace cortex;

        const int min_isize = 4;
        const int max_isize = 48;

        const int min_ksize = 3;
        const int max_ksize = 15;

        const int idims = 16;
        const int odims = 32;

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h",         "benchmark 3D convolutions");
        po_desc.add_options()("output",         "output");
        po_desc.add_options()("gparam",         "gradient wrt parameters (kernels)");
        po_desc.add_options()("ginput",         "gradient wrt inputs");

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

        const bool has_output = po_vm.count("output") > 0;
        const bool has_gparam = po_vm.count("gparam") > 0;
        const bool has_ginput = po_vm.count("ginput") > 0;

        // output
        if (has_output)
        {
                text::table_t table("size\\output [us]");
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
                                const auto header = make_header(idims, isize, ksize, odims);
                                test_config_output<tensor_t>(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(text::make_table_row_minimum_mark<size_t>());
                table.print(std::cout);
        }

        // gradient wrt parameters
        if (has_gparam)
        {
                text::table_t table("size\\gparam [us]");
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
                                const auto header = make_header(idims, isize, ksize, odims);
                                test_config_gparam<tensor_t>(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(text::make_table_row_minimum_mark<size_t>());
                table.print(std::cout);
        }

        // gradient wrt inputs
        if (has_ginput)
        {
                text::table_t table("size\\ginput [us]");
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
                                const auto header = make_header(idims, isize, ksize, odims);
                                test_config_ginput<tensor_t>(isize, idims, ksize, odims, table.append(header));
                        }
                }

                table.mark(text::make_table_row_minimum_mark<size_t>());
                table.print(std::cout);
        }

	return EXIT_SUCCESS;
}

