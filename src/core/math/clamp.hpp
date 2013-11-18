#ifndef NANOCV_CLAMP_H
#define NANOCV_CLAMP_H

#include <boost/algorithm/clamp.hpp>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // force values in a given range.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                using boost::algorithm::clamp;
                using boost::algorithm::clamp_range;
        }
}

#endif // NANOCV_CLAMP_H

