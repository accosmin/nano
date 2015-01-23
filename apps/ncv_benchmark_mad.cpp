#include "nanocv.h"
#include "util/mad.hpp"
#include "util/tabulator.h"
#include "tensor/mad.hpp"

using namespace ncv;

template
<
        typename top,
        typename tvector,
        typename tscalar = typename tvector::Scalar
>
tscalar test_mad(
        tabulator_t::row_t& row, top op, const tvector& vec1, const tvector& vec2, tscalar wei)
{
        vector_t cvec1 = vec1;
        vector_t cvec2 = vec2;

        const ncv::timer_t timer;

        op(cvec1.data(), wei, cvec1.size(), cvec2.data());

        row << timer.microseconds();

        return cvec2.sum();
}

template
<
        typename tscalar
>
void check(tscalar result, tscalar baseline, const char* name)
{
        const tscalar err = math::abs(result - baseline);
        if (!math::almost_equal(err, tscalar(0)))
        {
                std::cout << name << " FAILED (diff = " << err << ")!" << std::endl;
        }
}

void test_mad(size_t size, tabulator_t::row_t& row)
{
        vector_t vec1(size), vec2(size);
        vec1.setRandom();
        vec2.setRandom();

        scalar_t wei = vec1(0) + vec2(3);

        const scalar_t mad    = test_mad(row, ncv::mad<scalar_t>, vec1, vec2, wei);
        const scalar_t madul2 = test_mad(row, ncv::mad_unroll<scalar_t, 2>, vec1, vec2, wei);
        const scalar_t madul3 = test_mad(row, ncv::mad_unroll<scalar_t, 3>, vec1, vec2, wei);
        const scalar_t madul4 = test_mad(row, ncv::mad_unroll<scalar_t, 4>, vec1, vec2, wei);
        const scalar_t madul5 = test_mad(row, ncv::mad_unroll<scalar_t, 5>, vec1, vec2, wei);
        const scalar_t madul6 = test_mad(row, ncv::mad_unroll<scalar_t, 6>, vec1, vec2, wei);
        const scalar_t madul7 = test_mad(row, ncv::mad_unroll<scalar_t, 7>, vec1, vec2, wei);
        const scalar_t madul8 = test_mad(row, ncv::mad_unroll<scalar_t, 8>, vec1, vec2, wei);
        const scalar_t madeig = test_mad(row, ncv::tensor::mad_eig<scalar_t>, vec1, vec2, wei);

        check(mad,      mad, "mad");
        check(madul2,   mad, "madul2");
        check(madul3,   mad, "madul3");
        check(madul4,   mad, "madul4");
        check(madul5,   mad, "madul5");
        check(madul6,   mad, "madul6");
        check(madul7,   mad, "madul7");
        check(madul8,   mad, "madul8");
        check(madeig,   mad, "madeig");
}

int main(int argc, char* argv[])
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

        table.print(std::cout);

	return EXIT_SUCCESS;
}

