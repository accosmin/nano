#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_mad"

#include <boost/test/unit_test.hpp>
#include "libnanocv/types.h"
#include "libnanocv/util/mad.hpp"
#include "libnanocv/math/abs.hpp"
#include "libnanocv/math/epsilon.hpp"
#include "libnanocv/tensor/mad.hpp"

namespace test
{
        using namespace ncv;

        template
        <
                typename top,
                typename tvector,
                typename tscalar = typename tvector::Scalar
        >
        tscalar test_mad(top op, const tvector& vec1, const tvector& vec2, tscalar wei)
        {
                vector_t cvec1 = vec1;
                vector_t cvec2 = vec2;

                op(cvec1.data(), wei, cvec1.size(), cvec2.data());

                return cvec2.sum();
        }

        void test_mad(size_t size)
        {
                vector_t vec1(size), vec2(size);
                vec1.setRandom();
                vec2.setRandom();

                scalar_t wei = vec1(0) + vec2(3);

                const scalar_t mad    = test_mad(ncv::mad<scalar_t>, vec1, vec2, wei);
                const scalar_t madul2 = test_mad(ncv::mad_unroll<scalar_t, 2>, vec1, vec2, wei);
                const scalar_t madul3 = test_mad(ncv::mad_unroll<scalar_t, 3>, vec1, vec2, wei);
                const scalar_t madul4 = test_mad(ncv::mad_unroll<scalar_t, 4>, vec1, vec2, wei);
                const scalar_t madul5 = test_mad(ncv::mad_unroll<scalar_t, 5>, vec1, vec2, wei);
                const scalar_t madul6 = test_mad(ncv::mad_unroll<scalar_t, 6>, vec1, vec2, wei);
                const scalar_t madul7 = test_mad(ncv::mad_unroll<scalar_t, 7>, vec1, vec2, wei);
                const scalar_t madul8 = test_mad(ncv::mad_unroll<scalar_t, 8>, vec1, vec2, wei);
                const scalar_t madeig = test_mad(ncv::tensor::mad<scalar_t>, vec1, vec2, wei);

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                BOOST_CHECK_LE(math::abs(mad - mad), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul2), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul3), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul4), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul5), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul6), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul7), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madul8), epsilon);
                BOOST_CHECK_LE(math::abs(mad - madeig), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_mad)
{
        using namespace ncv;

        static const size_t min_size = 32 * 1024;
        static const size_t max_size = 4 * 1024 * 1024;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test::test_mad(size);
        }
}

