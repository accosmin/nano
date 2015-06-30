#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/tensor/conv3d.hpp"
#include <iostream>

using namespace ncv;

template
<
        typename top,
        typename ttensori,
        typename ttensork,
        typename ttensoro
>
static void test_cpu(tabulator_t::row_t& row, top op, const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
{
        const size_t trials = 1;
        row << ncv::measure_robustly_usec([&] ()
        {
                op(idata, kdata, odata);
        }, trials);
}

void test_conv3d(tabulator_t::row_t& row, int isize, int idims, int ksize, int odims)
{
        const int osize = isize - ksize + 1;
        const int kdims = odims * idims;

        tensor_t idata(idims, isize, isize);
        tensor_t kdata(kdims, ksize, ksize);
        tensor_t odata(odims, osize, osize);

        random_t<scalar_t> rng(-1.0, 1.0);

        idata.setRandom(rng);
        kdata.setRandom(rng);
        odata.setRandom(rng);

        idata.vector() /= isize;
        kdata.vector() /= ksize;
        odata.vector() /= osize;

        test_cpu(row, math::conv3d_output<tensor_t, tensor_t, tensor_t&>, idata, kdata, odata);
        test_cpu(row, tensor::conv3d_output<tensor_t, tensor_t, tensor_t&>, idata, kdata, odata);
}

int main(int, char* [])
{
        const int min_isize = 4;
        const int max_isize = 48;

        const int min_ksize = 3;
        const int max_ksize = 9;

        const int min_idims = 16;
        const int max_idims = 64;

        const int min_odims = 16;
        const int max_odims = 128;

        tabulator_t table("size\\method");
        table.header() << "dyn [us]"
                       << "toe [us]";

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                table.clear();

                for (int idims = min_idims; idims <= max_idims; idims *= 2)
                {
                        for (int ksize = min_ksize; ksize <= std::min(max_ksize, isize); ksize ++)
                        {
                                const int osize = isize - ksize + 1;

                                for (int odims = min_odims; odims <= max_odims; odims *= 2)
                                {
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
                        }
                }

                table.print(std::cout);

                std::cout << std::endl;
        }
                
	return EXIT_SUCCESS;
}

