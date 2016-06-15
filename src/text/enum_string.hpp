#pragma once

#include <map>
#include "stringi.h"

namespace nano
{
        ///
        /// \brief maps all possible values of an enum to string
        ///
        /// NB: to specialize it, such that nano::to_string & nano::from_string work on a particular enum
        ///
        template
        <
                typename tenum
        >
        std::map<tenum, string_t> enum_string();
}

