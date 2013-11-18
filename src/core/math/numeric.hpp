#ifndef NANOCV_NUMERIC_H
#define NANOCV_NUMERIC_H

#include <limits>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // numeric utility functions.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // square a value
                template
                <
                        typename tvalue
                >
                tvalue square(tvalue value)
                {
                        return value * value;
                }

                template
                <
                        typename tvalue
                >
                tvalue cube(tvalue value)
                {
                        return value * square(value);
                }

                // check two values if approximatively equal
                template
                <
                        typename tvalue
                >
                bool equal(tvalue value1, tvalue value2, tvalue epsilon = std::numeric_limits<tvalue>::epsilon())
                {
                        return value1 < value2 + epsilon && value2 < value1 + epsilon;
                }

                // sign value: x / |x|
                template
                <
                        typename tvalue,
                        typename tresult
                >
                tresult sign(tvalue value)
                {
                        static const tvalue zero = static_cast<tvalue>(0);
                        return  value > zero ? static_cast<tresult>(1) :
                                (value < zero ? static_cast<tresult>(-1) : static_cast<tresult>(0));
                }

                // kronocker: 1 (if true), 0 (else)
                template
                <
                        typename tresult
                >
                tresult kronocker(bool condition)
                {
                        return condition ? static_cast<tresult>(1) : static_cast<tresult>(0);
                }
        }
}

#endif // NANOCV_NUMERIC_H

