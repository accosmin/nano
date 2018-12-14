#include "utest.h"
#include "core/numeric.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/wlearner_stump.h"

using namespace nano;

static auto get_outputs()
{
        tensor4d_t outputs(2, 3, 1, 1);
        outputs.tensor(0).constant(-1);
        outputs.tensor(1).constant(+1);
        return outputs;
}

const auto outputs = get_outputs();

UTEST_BEGIN_MODULE(test_model_stump)

UTEST_CASE(getset)
{
        wlearner_real_stump_t learner;
        learner.feature(2);
        learner.threshold(scalar_t(-2.5));
        learner.outputs(outputs);

        UTEST_CHECK_EQUAL(learner.feature(), 2);
        UTEST_CHECK_EQUAL(learner.threshold(), scalar_t(-2.5));
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(), outputs.array(), epsilon0<scalar_t>());
}

UTEST_CASE(output)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        wlearner_real_stump_t learner;
        learner.feature(0);
        learner.threshold(scalar_t(-2.5));
        learner.outputs(outputs);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2)).size(), 3);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0))(0), +1);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1))(0), -1);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2))(0), -1);
}

UTEST_CASE(scale1)
{
        wlearner_real_stump_t learner;
        learner.feature(0);
        learner.threshold(scalar_t(-3.5));
        learner.outputs(outputs);

        learner.scale(scalar_t(0.3));

        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(), outputs.array() * scalar_t(0.3), epsilon0<scalar_t>());
}

UTEST_CASE(scalex)
{
        wlearner_real_stump_t learner;
        learner.feature(0);
        learner.threshold(scalar_t(-3.5));
        learner.outputs(outputs);

        vector_t factors(3);
        factors(0) = 0.1;
        factors(1) = 0.2;
        factors(2) = 0.3;
        learner.scale(factors);

        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(0), outputs.array(0) * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(1), outputs.array(1) * factors.array(), epsilon0<scalar_t>());
}

UTEST_CASE(serialize)
{
        wlearner_real_stump_t learner;
        learner.feature(4);
        learner.threshold(scalar_t(-3.5));
        learner.outputs(outputs);

        const auto path_learner = "learner.stump";
        {
                obstream_t ostream(path_learner);
                UTEST_CHECK(learner.save(ostream));
        }
        {
                wlearner_real_stump_t learner2;

                ibstream_t istream(path_learner);
                UTEST_CHECK(learner2.load(istream));

                UTEST_CHECK_EQUAL(learner.feature(), learner2.feature());
                UTEST_CHECK_EQUAL(learner.threshold(), learner2.threshold());
                UTEST_CHECK_EIGEN_CLOSE(learner.outputs().vector(), learner2.outputs().vector(), epsilon0<scalar_t>());
        }

        std::remove(path_learner);
}

// todo: check fitting, computing the fvalues and the threshold

UTEST_END_MODULE()
