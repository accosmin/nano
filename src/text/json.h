#pragma once

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
                to_json(json, name, value);
                to_json(json, nvs...);
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
                        value = json[name].get<tvalue>();
                }
                from_json(json, nvs...);
        }
}
