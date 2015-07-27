#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/conv3d.hpp"
#include <iostream>

using namespace ncv;

namespace
{
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
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_output(op, idata, kdata, odata);
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
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_ginput(op, idata, kdata, odata);
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
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_gparam(op, idata, kdata, odata);
                }, trials);
        }

        void test_config(int isize, int idims, int ksize, int odims,
                tabulator_t::row_t& row_output,
                tabulator_t::row_t& row_ginput,
                tabulator_t::row_t& row_gparam)
        {
                const int osize = isize - ksize + 1;
                const int kdims = odims * idims;

                random_t<scalar_t> rng(-1.0 / isize, 1.0 / isize);

                tensor_t idata(idims, isize, isize);
                tensor_t kdata(kdims, ksize, ksize);
                tensor_t odata(odims, osize, osize);

                idata.setRandom(rng);
                kdata.setRandom(rng);
                odata.setRandom(rng);

                tensor_t idata_ret = idata;
                tensor_t kdata_ret = kdata;
                tensor_t odata_ret = odata;

                tensor::conv3d_t<tensor_t> conv3d;
                conv3d.reset(kdata, idims, odims);

                const size_t trials = 16;

                // output
                row_output << measure_output(math::conv2d_eig_t(), idata, kdata, odata_ret, trials);
                row_output << measure_output(math::conv2d_cpp_t(), idata, kdata, odata_ret, trials);
                row_output << measure_output(math::conv2d_dot_t(), idata, kdata, odata_ret, trials);
                row_output << measure_output(math::conv2d_mad_t(), idata, kdata, odata_ret, trials);
                row_output << measure_output(math::conv2d_dyn_t(), idata, kdata, odata_ret, trials);
                row_output << ncv::measure_robustly_usec([&] () { conv3d.output(idata, odata_ret); }, trials);

                // gradient wrt input
                row_ginput << measure_ginput(ncv::math::corr2d_egb_t(), idata_ret, kdata, odata, trials);
                row_ginput << measure_ginput(ncv::math::corr2d_egr_t(), idata_ret, kdata, odata, trials);
                row_ginput << measure_ginput(ncv::math::corr2d_cpp_t(), idata_ret, kdata, odata, trials);
                row_ginput << measure_ginput(ncv::math::corr2d_mdk_t(), idata_ret, kdata, odata, trials);
                row_ginput << measure_ginput(ncv::math::corr2d_mdo_t(), idata_ret, kdata, odata, trials);
                row_ginput << measure_ginput(ncv::math::corr2d_dyn_t(), idata_ret, kdata, odata, trials);
                row_ginput << ncv::measure_robustly_usec([&] () { conv3d.ginput(idata_ret, odata); }, trials);

                // gradient wrt parameters
                row_gparam << measure_gparam(math::conv2d_eig_t(), idata, kdata_ret, odata, trials);
                row_gparam << measure_gparam(math::conv2d_cpp_t(), idata, kdata_ret, odata, trials);
                row_gparam << measure_gparam(math::conv2d_dot_t(), idata, kdata_ret, odata, trials);
                row_gparam << measure_gparam(math::conv2d_mad_t(), idata, kdata_ret, odata, trials);
                row_gparam << measure_gparam(math::conv2d_dyn_t(), idata, kdata_ret, odata, trials);
                row_gparam << ncv::measure_robustly_usec([&] () { conv3d.gparam(idata, kdata_ret, odata); }, trials);
        }
}

int main(int, char* [])
{
        const int min_isize = 8;
        const int max_isize = 48;

        const int min_ksize = 3;
        const int max_ksize = 15;

        const int idims = 16;
        const int odims = 32;

        tabulator_t table_output("size\\output [us]");
        table_output.header()
                << "2D (eig)"
                << "2D (cpp)"
                << "2D (dot)"
                << "2D (mad)"
                << "2D (dyn)"
                << "3D (lin)";

        tabulator_t table_ginput("size\\ginput [us]");
        table_ginput.header()
                << "2D (egb)"
                << "2D (egr)"
                << "2D (cpp)"
                << "2D (mkd)"
                << "2D (mko)"
                << "2D (dyn)"
                << "3D (lin)";

        tabulator_t table_gparam("size\\gparam [us]");
        table_gparam.header()
                << "2D (eig)"
                << "2D (cpp)"
                << "2D (dot)"
                << "2D (mad)"
                << "2D (dyn)"
                << "3D (lin)";

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                table_output.clear();
                table_ginput.clear();
                table_gparam.clear();

                for (int ksize = min_ksize; ksize <= std::min(isize - min_ksize, max_ksize); ksize += 2)
                {
                        const int osize = isize - ksize + 1;

                        const string_t header =
                                "(" +
                                text::to_string(idims) + "x" +
                                text::to_string(isize) + "x" +
                                text::to_string(isize) + " @ " +
                                text::to_string(ksize) + "x" +
                                text::to_string(ksize) + " -> " +
                                text::to_string(odims) + "x" +
                                text::to_string(osize) + "x" +
                                text::to_string(osize) +
                                ")";

                        test_config(isize, idims, ksize, odims,
                                    table_output.append(header),
                                    table_ginput.append(header),
                                    table_gparam.append(header));
                }

                table_output.print(std::cout);
                table_ginput.print(std::cout);
                table_gparam.print(std::cout);

                std::cout << std::endl;
        }

	return EXIT_SUCCESS;
}

