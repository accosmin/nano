#pragma once

#include <map>
#include "stringi.h"
#include "math/stats.h"

namespace nano
{
        /// <entity, timing statistics in microseconds>
        using timing_t = stats_t<size_t>;
        using timings_t = std::map<string_t, timing_t>;
}
