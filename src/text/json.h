#pragma once

#include "cast.h"
#include <nlohmann/json.hpp>

namespace nano
{
        using json_t = nlohmann::json;
        using jsons_t = std::vector<json_t>;

        ///
        /// \brief serialize attributes to JSON.
        ///
        inline void to_json(json_t&)
        {
        }

        template <typename tvalue, typename... tothers>
        void to_json(json_t& json, const char* name, const tvalue value, tothers&&... nvs)
        {
                json[name] = to_string(value);
                to_json(json, nvs...);
        }

        template <typename... tattributes>
        json_t to_json(tattributes&&... attributes)
        {
                json_t json;
                to_json(json, attributes...);
                return json;
        }

        ///
        /// \brief deserialize attributes from JSON if present.
        ///
        inline void from_json(const json_t&)
        {
        }

        template <typename tvalue, typename... tothers>
        void from_json(const json_t& json, const char* name, tvalue& value, tothers&&... nvs)
        {
                if (json.count(name))
                {
                        value = from_string<tvalue>(json[name].get<string_t>());
                }
                from_json(json, nvs...);
        }
}
