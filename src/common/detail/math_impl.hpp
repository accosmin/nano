#pragma once

#include <cmath>

namespace ncv
{
        namespace math
        {
                // implementation detail
                namespace detail
                {
                        template
                        <
                                typename tround,
                                bool tround_integral,
                                typename tvalue,
                                bool tvalue_integral
                        >
                        struct cast
                        {
                                static tround dispatch(tvalue value)
                                {
                                        return static_cast<tround>(value);
                                }
                        };

                        template
                        <
                                typename tround,
                                typename tvalue
                        >
                        struct cast<tround, true, tvalue, false>
                        {
                                static tround dispatch(tvalue value)
                                {
                                        return static_cast<tround>(std::nearbyint(value));
                                }
                        };
                }
        }
}

