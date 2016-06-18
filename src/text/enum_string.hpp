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
        template <typename tenum>
        std::map<tenum, string_t> enum_string();

        ///
        /// \brief collect the names for each enum value
        ///
        template <typename tenum>
        strings_t enum_strings()
        {
                strings_t ret;
                for (const auto& elem : enum_string<tenum>())
                {
                        ret.push_back(elem.second);
                }
                return ret;
        }
}

