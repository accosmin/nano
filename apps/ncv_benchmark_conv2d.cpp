#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/conv2d.hpp"
#include <iostream>

using namespace ncv;

const size_t trials = 16;

template
<
        typename top,
        typename tmatrix,
        typename tscalar = typename tmatrix::Scalar
>
static void test_cpu(tabulator_t::row_t& row, top op, const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
{
        row << ncv::measure_robustly_usec([&] ()
        {
                odata.setZero();
                op(idata, kdata, odata);
        }, trials);
}

void test_conv2d(tabulator_t::row_t& row, int isize, int ksize)
{
        const int osize = isize - ksize + 1;

        matrix_t idata(isize, isize);
        matrix_t kdata(ksize, ksize);
        matrix_t odata(osize, osize);

        idata.setRandom();
        kdata.setRandom();
        odata.setRandom();

        idata /= isize;
        kdata /= ksize;
        odata /= osize;

        test_cpu(row, ncv::math::conv2d_eig<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_cpp<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_dot<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_mad<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_dyn<matrix_t>, idata, kdata, odata);
}

int main(int, char* [])
{
        const int min_isize = 12;
        const int max_isize = 48;
        const int min_ksize = 3;

        tabulator_t table("size\\method");
        table.header() << "eig [us]"
                        << "cpp [us]"
                        << "dot [us]"
                        << "mad [us]"
                        << "dyn [us]";

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                table.clear();

                for (int ksize = min_ksize; ksize <= isize - min_ksize; ksize += 2)
                {
                        const string_t header = "(" +
                                text::to_string(isize) + "x" + text::to_string(isize) + "@" +
                                text::to_string(ksize) + "x" + text::to_string(ksize) + ")";

                        tabulator_t::row_t& row = table.append(header);

                        test_conv2d(row, isize, ksize);
                }

                table.print(std::cout);

                std::cout << std::endl;
        }
                
	return EXIT_SUCCESS;
}

