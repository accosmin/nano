#pragma once

#include <map>
#include <string>

namespace text
{
        ///
        /// \brief maps all possible values of an enum to string
        ///
        /// NB: to specialize it, such that text::to_string & text::from_string work a particular enum
        ///
        template
        <
                typename tenum
        >
        std::map<tenum, std::string> enum_string();
}

