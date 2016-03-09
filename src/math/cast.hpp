#pragma once

#include <cmath>
#include <type_traits>

namespace zob
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
                                return static_cast<tround>(std::round(value));
                        }
                };
        }

        ///
        /// \brief cast a value to another type (with rounding to the closest if necessary)
        ///
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
}

