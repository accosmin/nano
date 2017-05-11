#pragma once

#include <map>
#include "stringi.h"

namespace nano
{
        ///
        /// \brief maps all possible values of an enum to string.
        ///
        /// NB: to specialize it, such that nano::to_string & nano::from_string work on a particular enum
        ///
        template <typename tenum>
        std::map<tenum, std::string> enum_string();

        ///
        /// \brief collect all the values for an enum type.
        ///
        template <typename tenum>
        std::vector<tenum> enum_values()
        {
                const auto mapping = enum_string<tenum>();

                std::vector<tenum> ret;
                for (const auto& elem : mapping)
                {
                        ret.push_back(elem.first);
                }
                return ret;
        }
}
