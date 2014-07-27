#ifndef NANOCV_CAST_H
#define NANOCV_CAST_H

#include <type_traits>
#include <boost/algorithm/clamp.hpp>

namespace ncv
{
        namespace math
        {
                // forward boost functions
                using boost::algorithm::clamp;
                using boost::algorithm::clamp_range;

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

                // cast a value to another type (with rounding to the closest if necessary)
                template
                <
                        typename tround,
                        typename tvalue
                >
                tround cast(tvalue value)
                {
                        return  detail::cast<
                                tround, std::is_integral<tround>::value,
                                tvalue, std::is_integral<tvalue>::value>::dispatch(value);
                }

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
        }
}

#endif // NANOCV_CAST_H

