#include "color.h"
#include "text/to_string.hpp"

namespace nano
{
        std::ostream& operator<<(std::ostream& os, color_mode mode)
        {
                return os << nano::to_string(mode);
        }
}
