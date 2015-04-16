#include "nanocv/tensor.h"
#include "nanocv/string.h"
#include "nanocv/measure.hpp"
#include "nanocv/tabulator.h"
#include "nanocv/placeholders.h"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/tensor/conv2d.hpp"
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

        const matrix_t tdata = ncv::tensor::conv2d_make_toeplitz(idata, kdata, odata);
        
        test_cpu(row, ncv::math::conv2d_eig<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_cpp<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_dot<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_mad<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::math::conv2d_dyn<matrix_t>, idata, kdata, odata);
        test_cpu(row, ncv::tensor::conv2d_toeplitz<matrix_t>, idata, kdata, odata);
        test_cpu(row, std::bind(ncv::tensor::conv2d_toeplitz_buffered<matrix_t>, ncv::_1, ncv::_2, std::cref(tdata), _3), idata, kdata, odata);
}

int main(int argc, char* argv[])
{
        const int min_isize = 24;
        const int max_isize = 48;
        const int min_ksize = 5;

        tabulator_t table("size\\method");
        table.header() << "eig [us]"
                        << "cpp [us]"
                        << "dot [us]"
                        << "mad [us]"
                        << "dyn [us]"
                        << "toe [us]"
                        << "toe (buff) [us]";

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

