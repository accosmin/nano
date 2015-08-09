#include "nanocv/tensor.h"
#include "nanocv/tabulator.h"
#include "nanocv/measure.hpp"
#include "nanocv/math/mad.hpp"
#include "nanocv/tensor/mad.hpp"
#include <iostream>

using namespace ncv;

template
<
        typename top,
        typename tvector,
        typename tscalar = typename tvector::Scalar
>
static void test_mad(tabulator_t::row_t& row, top op, const tvector& vec1, const tvector& vec2, tscalar wei)
{
        vector_t cvec1 = vec1;
        vector_t cvec2 = vec2;

        const size_t trials = 16;

        row << ncv::measure_robustly_usec([&] ()
        {
                op(cvec1.data(), wei, cvec1.size(), cvec2.data());
        }, trials);
}

static void test_mad(size_t size, tabulator_t::row_t& row)
{
        vector_t vec1(size), vec2(size);
        vec1.setRandom();
        vec2.setRandom();

        scalar_t wei = vec1(0) + vec2(3);

        test_mad(row, ncv::math::mad<scalar_t>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 2>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 3>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 4>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 5>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 6>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 7>, vec1, vec2, wei);
        test_mad(row, ncv::math::mad_unroll<scalar_t, 8>, vec1, vec2, wei);
        test_mad(row, ncv::tensor::mad<scalar_t>, vec1, vec2, wei);
}

int main(int, char* [])
{
        static const size_t min_size = 32 * 1024;
        static const size_t max_size = 4 * 1024 * 1024;

        tabulator_t table("size\\mad");

        table.header() << "mad [us]"
                       << "madul2 [us]"
                       << "madul3 [us]"
                       << "madul4 [us]"
                       << "madul5 [us]"
                       << "madul6 [us]"
                       << "madul7 [us]"
                       << "madul8 [us]"
                       << "madeig [us]";

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                tabulator_t::row_t& row = table.append(text::to_string(size / 1024) + "K");

                test_mad(size, row);
        }

        table.mark_min_number();
        table.print(std::cout);

	return EXIT_SUCCESS;
}

