#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/math/random.hpp"
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
                const ttensori& idata, const ttensork&& kdata, const ttensoro& odata, const size_t trials = 16)
        {
                return ncv::measure_robustly_usec([&] ()
                {
                        math::conv3d_gparam(op, idata, kdata, odata);
                }, trials);
        }

        void test_corr2d(int isize, int idims, int ksize, int odims,
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

                // output
                row_output << measure_output(math::conv2d_eig_t(), idata, kdata, odata_ret);
                row_output << measure_output(math::conv2d_cpp_t(), idata, kdata, odata_ret);
                row_output << measure_output(math::conv2d_dot_t(), idata, kdata, odata_ret);
                row_output << measure_output(math::conv2d_mad_t(), idata, kdata, odata_ret);
                row_output << measure_output(math::conv2d_dyn_t(), idata, kdata, odata_ret);

                // gradient wrt input
                row_ginput << measure_ginput(ncv::math::corr2d_egb_t(), idata_ret, kdata, odata);
                row_ginput << measure_ginput(ncv::math::corr2d_egr_t(), idata_ret, kdata, odata);
                row_ginput << measure_ginput(ncv::math::corr2d_cpp_t(), idata_ret, kdata, odata);
                row_ginput << measure_ginput(ncv::math::corr2d_mdk_t(), idata_ret, kdata, odata);
                row_ginput << measure_ginput(ncv::math::corr2d_mdo_t(), idata_ret, kdata, odata);
                row_ginput << measure_ginput(ncv::math::corr2d_dyn_t(), idata_ret, kdata, odata);

                // gradient wrt parameters
                row_gparam << measure_gparam(math::conv2d_eig_t(), idata, kdata_ret, odata);
                row_gparam << measure_gparam(math::conv2d_cpp_t(), idata, kdata_ret, odata);
                row_gparam << measure_gparam(math::conv2d_dot_t(), idata, kdata_ret, odata);
                row_gparam << measure_gparam(math::conv2d_mad_t(), idata, kdata_ret, odata);
                row_gparam << measure_gparam(math::conv2d_dyn_t(), idata, kdata_ret, odata);
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

        tabulator_t table("size\\method");
        table.header() << "egb [us]"
                       << "egr [us]"
                       << "cpp [us]"
                       << "mkd [us]"
                       << "mko [us]"
                       << "dyn [us]";

        table.header() << "eig [us]"
                       << "cpp [us]"
                       << "dot [us]"
                       << "mad [us]"
                       << "dyn [us]";

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                table.clear();

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

                        tabulator_t::row_t& row = table.append(header);

                        test_corr2d(row, isize, idims, ksize, odims);
                }

                table.print(std::cout);

                std::cout << std::endl;
        }

	return EXIT_SUCCESS;
}

