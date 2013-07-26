#ifndef NANOCV_NUMERIC_H
#define NANOCV_NUMERIC_H

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // numeric utility functions.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        }
}

#endif // NANOCV_NUMERIC_H

