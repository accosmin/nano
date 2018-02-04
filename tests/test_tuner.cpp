#include "utest.h"
#include "tuner.h"
#include "math/epsilon.h"
#include "text/json_reader.h"

using namespace nano;

static void get(const string_t& json, scalar_t& x)
{
        x = 100;
        json_reader_t reader(json);
        reader.object("x", x);
}

static void get(const string_t& json, scalar_t& x, scalar_t& y)
{
        x = y = 100;
        json_reader_t reader(json);
        reader.object("x", x, "y", y);
}

static void get(const string_t& json, scalar_t& x, scalar_t& y, scalar_t& z)
{
        x = y = z = 100;
        json_reader_t reader(json);
        reader.object("x", x, "y", y, "z", z);
}

NANO_BEGIN_MODULE(test_tuner)

NANO_CASE(tuner1d)
{
        const auto eval = [] (const scalar_t x)
        {
                return (x - 1) * (x - 1);
        };

        tuner_t tuner;
        tuner.add_linear("x", -3, +3);

        scalar_t x = 0;
        for (int trial = 0; trial < 100; ++ trial)
        {
                get(tuner.get(), x);

                NANO_CHECK_LESS(x, scalar_t(+3));
                NANO_CHECK_GREATER(x, scalar_t(-3));

                tuner.score(1 / (1 + eval(x)));
        }

        //get(tuner.optimum(), x);
        //NANO_CHECK_LESS(eval(x), epsilon2<scalar_t>());
        //NANO_CHECK_CLOSE(x, 1, epsilon3<scalar_t>());
}

NANO_CASE(tuner2d)
{
        const auto eval = [] (const scalar_t x, const scalar_t y)
        {
                return (x - 1) * (x - 1) + std::fabs(y - 2);
        };

        tuner_t tuner;
        tuner.add_linear("x", -3, +3);
        tuner.add_base10("y", 0, 1);

        scalar_t x = 0, y = 0;
        for (int trial = 0; trial < 100; ++ trial)
        {
                get(tuner.get(), x, y);

                NANO_CHECK_LESS(x, scalar_t(+3));
                NANO_CHECK_LESS(y, scalar_t(10));
                NANO_CHECK_GREATER(x, scalar_t(-3));
                NANO_CHECK_GREATER(y, scalar_t(1));

                tuner.score(1 / (1 + eval(x, y)));
        }

        //get(tuner.optimum(), x, y);
        //NANO_CHECK_LESS(eval(x, y), epsilon2<scalar_t>());
        //NANO_CHECK_CLOSE(x, 1, epsilon3<scalar_t>());
        //NANO_CHECK_CLOSE(y, 2, epsilon3<scalar_t>());
}

NANO_CASE(tuner3d)
{
        const auto eval = [] (const scalar_t x, const scalar_t y, const scalar_t z)
        {
                return (x - 1) * (x - 1) + std::fabs(y - 2) + (z - x) * (z - x);
        };

        tuner_t tuner;
        tuner.add_linear("x", -2, +2);
        tuner.add_base10("y", 0, 1);
        tuner.add_finite("z", make_scalars(-2, 0, 1));

        scalar_t x = 0, y = 0, z = 0;
        for (int trial = 0; trial < 100; ++ trial)
        {
                get(tuner.get(), x, y, z);

                NANO_CHECK_LESS(x, scalar_t(+3));
                NANO_CHECK_LESS(y, scalar_t(10));
                NANO_CHECK_GREATER(x, scalar_t(-3));
                NANO_CHECK_GREATER(y, scalar_t(1));
                NANO_CHECK_CLOSE((z + 2) * z * (z - 1), scalar_t(0), epsilon0<scalar_t>());

                tuner.score(1 / (1 + eval(x, y, z)));
        }

        //get(tuner.optimum(), x, y, z);
        //NANO_CHECK_LESS(eval(x, y, z), epsilon3<scalar_t>());
        //NANO_CHECK_CLOSE(x, 1, epsilon3<scalar_t>());
        //NANO_CHECK_CLOSE(y, 2, epsilon3<scalar_t>());
        //NANO_CHECK_CLOSE(z, x, epsilon3<scalar_t>());
}

NANO_END_MODULE()
