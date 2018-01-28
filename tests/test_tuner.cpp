#include "utest.h"
#include "tuner.h"
#include "text/json_reader.h"

#include <iostream>

using namespace nano;

NANO_BEGIN_MODULE(test_tuner)

NANO_CASE(tuner1d)
{
        const auto eval = [] (const scalar_t x)
        {
                return (x - 1) * (x - 1);
        };

        tuner_t tuner;
        tuner.add("x", -5, +5, 1);

        for (int trial = 0; trial < 100; ++ trial)
        {
                const auto config = tuner.get();
                json_reader_t reader(config);
                scalar_t x = 0;
                reader.object("x", x);

                const auto xeval = eval(x);
                const auto score = 1 / (1 + xeval);
                tuner.score(score);
                std::cout << "x = " << x << ", f(x) = " << xeval << ", score = " << score << std::endl;
        }
}

NANO_END_MODULE()
