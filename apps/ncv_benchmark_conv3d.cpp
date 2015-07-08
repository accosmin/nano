#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/tensor/conv3d.hpp"
#include <iostream>

using namespace ncv;

void test_conv3d(tabulator_t::row_t& row, int isize, int idims, int ksize, int odims)
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

        tensor_t idata_dyn = idata, idata_lin = idata;
        tensor_t kdata_dyn = kdata, kdata_lin = kdata;
        tensor_t odata_dyn = odata, odata_lin = odata;

        tensor::conv3d_t<tensor_t> conv3d;
        conv3d.reset(kdata, idims, odims);

        const size_t trials = 1;

        // 2D convolution-based
        row << ncv::measure_robustly_usec([&] ()
        {
                math::conv3d_output(math::conv2d_dyn_t(), idata, kdata, odata_dyn);
        }, trials);
        row << ncv::measure_robustly_usec([&] ()
        {
                math::conv3d_gparam(math::conv2d_dyn_t(), idata, kdata_dyn, odata);
        }, trials);
        row << ncv::measure_robustly_usec([&] ()
        {
                math::conv3d_ginput(math::corr2d_dyn_t(), idata_dyn, kdata, odata);
        }, trials);

        // linearized tensors-based
        row << ncv::measure_robustly_usec([&] ()
        {
                conv3d.output(idata, odata_lin);
        }, trials);
        row << ncv::measure_robustly_usec([&] ()
        {
                conv3d.gparam(idata, kdata_lin, odata);
        }, trials);
        row << ncv::measure_robustly_usec([&] ()
        {
                conv3d.ginput(idata_lin, odata);
        }, trials);
}

int main(int, char* [])
{
        const int min_isize = 4;
        const int max_isize = 48;

        const int min_ksize = 1;
        const int max_ksize = 9;

        const int idims = 16;
        const int odims = 32;

        tabulator_t table("size\\method");
        table.header() << "dyn (odata) [us]"
                       << "dyn (kdata) [us]"
                       << "dyn (idata) [us]"
                       << "lin (odata) [us]"
                       << "lin (kdata) [us]"
                       << "lin (idata) [us]";

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                table.clear();

                for (int ksize = min_ksize; ksize <= std::min(max_ksize, isize); ksize ++)
                {
                        const int osize = isize - ksize + 1;

                        const string_t header = "(" +
                                                text::to_string(idims) + "x" +
                                                text::to_string(isize) + "x" +
                                                text::to_string(isize) + " @ " +
                                                text::to_string(ksize) + "x" +
                                                text::to_string(ksize) + " -> " +
                                                text::to_string(odims) + "x" +
                                                text::to_string(osize) + "x" +
                                                text::to_string(osize) + ")";

                        tabulator_t::row_t& row = table.append(header);

                        test_conv3d(row, isize, idims, ksize, odims);
                }

                table.print(std::cout);

                std::cout << std::endl;
        }
                
	return EXIT_SUCCESS;
}

