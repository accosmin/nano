#include "utest.h"
#include "core/numeric.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/wlearner_linear.h"

using namespace nano;

static auto get_a()
{
        tensor3d_t x(3, 1, 1);
        x.constant(-1);
        return x;
}

static auto get_b()
{
        tensor3d_t x(3, 1, 1);
        x.constant(+1);
        return x;
}

const auto a = get_a();
const auto b = get_b();

UTEST_BEGIN_MODULE(test_wlearner_linear)

UTEST_CASE(getset)
{
        wlearner_linear_t learner;
        learner.feature(2);
        learner.a(a);
        learner.b(b);

        UTEST_CHECK_EQUAL(learner.feature(), 2);
        UTEST_CHECK_EIGEN_CLOSE(learner.a().array(), a.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.b().array(), b.array(), epsilon0<scalar_t>());
}

UTEST_CASE(output)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2)).size(), 3);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0))(0), +3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1))(0), +4);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2))(0), +5);
}

UTEST_CASE(scale1)
{
        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        learner.scale(scalar_t(0.3));

        UTEST_CHECK_EIGEN_CLOSE(learner.a().array(), a.array() * scalar_t(0.3), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.b().array(), b.array() * scalar_t(0.3), epsilon0<scalar_t>());
}

UTEST_CASE(scalex)
{
        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        vector_t factors(3);
        factors(0) = 0.1;
        factors(1) = 0.2;
        factors(2) = 0.3;
        learner.scale(factors);

        UTEST_CHECK_EIGEN_CLOSE(learner.a().array(), a.array() * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.b().array(), b.array() * factors.array(), epsilon0<scalar_t>());
}

UTEST_CASE(serialize)
{
        wlearner_linear_t learner;
        learner.feature(4);
        learner.a(a);
        learner.b(b);

        const auto path_learner = "learner.linear";
        {
                obstream_t ostream(path_learner);
                UTEST_CHECK(learner.save(ostream));
        }
        {
                wlearner_linear_t learner2;

                ibstream_t istream(path_learner);
                UTEST_CHECK(learner2.load(istream));

                UTEST_CHECK_EQUAL(learner.feature(), learner2.feature());
                UTEST_CHECK_EIGEN_CLOSE(learner.a().vector(), learner2.a().vector(), epsilon0<scalar_t>());
                UTEST_CHECK_EIGEN_CLOSE(learner.b().vector(), learner2.b().vector(), epsilon0<scalar_t>());
        }

        std::remove(path_learner);
}

UTEST_CASE(serialize_invalid_path)
{
        {
                wlearner_linear_t learner;
                obstream_t ostream("/tmp2/x2/y2/file");
                UTEST_CHECK(!learner.save(ostream));
        }
        {
                wlearner_linear_t learner;
                ibstream_t istream("/tmp2/x2/y2/file");
                UTEST_CHECK(!learner.load(istream));
        }
}

// todo: check fitting

UTEST_END_MODULE()
