#pragma once

namespace ncv
{
        namespace math
        {
                ///
                /// \brief clamp value in the [min_value, max_value] range
                ///
                template
                <
                        typename tscalar,
                        typename tscalar_min,
                        typename tscalar_max
                >
                tscalar clamp(tscalar value, tscalar_min min_value, tscalar_max max_value)
                {
                        return  value < static_cast<tscalar>(min_value) ? static_cast<tscalar>(min_value) :
                                (value > static_cast<tscalar>(max_value) ? static_cast<tscalar>(max_value) : value);
                }
        }
}

