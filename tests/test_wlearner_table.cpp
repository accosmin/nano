#include <utest/utest.h>
#include "core/numeric.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/wlearner_table.h"

using namespace nano;

static auto get_outputs()
{
        tensor4d_t o(5, 3, 1, 1);
        o.tensor(0).constant(-1);
        o.tensor(1).constant(+1);
        o.tensor(2).constant(+3);
        o.tensor(3).constant(+5);
        o.tensor(4).constant(+7);
        return o;
}

const auto outputs = get_outputs();

UTEST_BEGIN_MODULE(test_wlearner_table)

UTEST_CASE(getset)
{
        wlearner_real_table_t learner;
        learner.feature(2);
        learner.outputs(outputs);

        UTEST_CHECK_EQUAL(learner.feature(), 2);
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(), outputs.array(), epsilon0<scalar_t>());
}

UTEST_CASE(output)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, +0, +5);
        inputs.vector(1) = vector_t::LinSpaced(6, +1, +6);
        inputs.vector(2) = vector_t::LinSpaced(6, +2, +7);

        wlearner_real_table_t learner;
        learner.feature(2);
        learner.outputs(outputs);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1)).size(), 3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2)).size(), 3);

        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(0))(0), +3);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(1))(0), +5);
        UTEST_CHECK_EQUAL(learner.output(inputs.tensor(2))(0), +7);
}

UTEST_CASE(scale1)
{
        wlearner_real_table_t learner;
        learner.feature(0);
        learner.outputs(outputs);

        learner.scale(scalar_t(0.3));

        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(), outputs.array() * scalar_t(0.3), epsilon0<scalar_t>());
}

UTEST_CASE(scalex)
{
        wlearner_real_table_t learner;
        learner.feature(0);
        learner.outputs(outputs);

        vector_t factors(3);
        factors(0) = 0.1;
        factors(1) = 0.2;
        factors(2) = 0.3;
        learner.scale(factors);

        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(0), outputs.array(0) * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(1), outputs.array(1) * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(2), outputs.array(2) * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(3), outputs.array(3) * factors.array(), epsilon0<scalar_t>());
        UTEST_CHECK_EIGEN_CLOSE(learner.outputs().array(4), outputs.array(4) * factors.array(), epsilon0<scalar_t>());
}

UTEST_CASE(serialize)
{
        wlearner_discrete_table_t learner;
        learner.feature(4);
        learner.outputs(outputs);

        const auto path_learner = "learner.table";
        {
                obstream_t ostream(path_learner);
                UTEST_CHECK(learner.save(ostream));
        }
        {
                wlearner_discrete_table_t learner2;

                ibstream_t istream(path_learner);
                UTEST_CHECK(learner2.load(istream));

                UTEST_CHECK_EQUAL(learner.feature(), learner2.feature());
                UTEST_CHECK_EIGEN_CLOSE(learner.outputs().vector(), learner2.outputs().vector(), epsilon0<scalar_t>());
        }

        std::remove(path_learner);
}

UTEST_CASE(serialize_wrong_type)
{
        wlearner_discrete_table_t learner;
        learner.feature(4);
        learner.outputs(outputs);

        const auto path_learner = "learner.table";
        {
                obstream_t ostream(path_learner);
                UTEST_CHECK(learner.save(ostream));
        }
        {
                wlearner_real_table_t learner2;

                ibstream_t istream(path_learner);
                UTEST_CHECK(!learner2.load(istream));
        }

        std::remove(path_learner);
}

UTEST_CASE(serialize_invalid_path)
{
        {
                wlearner_real_table_t learner;
                obstream_t ostream("/tmp2/x2/y2/file");
                UTEST_CHECK(!learner.save(ostream));
        }
        {
                wlearner_discrete_table_t learner;
                ibstream_t istream("/tmp2/x2/y2/file");
                UTEST_CHECK(!learner.load(istream));
        }
}

// todo: check fitting

UTEST_END_MODULE()
