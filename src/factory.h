#pragma once

#include <map>
#include <regex>
#include <memory>
#include <stdexcept>
#include <functional>
#include "configurable.h"

namespace nano
{
        ///
        /// \brief create objects of similar type that implement the configable_t interface.
        ///
        template <typename tobject>
        struct factory_t
	{
                using trobject = std::unique_ptr<tobject>;
                using tmaker = std::function<trobject(const string_t&)>;

                ///
                /// \brief register a new object with the given ID.
                ///
                template <typename tobject_impl>
                bool add(const string_t& id, const string_t& description);

                ///
                /// \brief check if an object was registered with the given ID.
                ///
                bool has(const string_t& id) const;

                ///
                /// \brief retrieve the object with the given ID, constructed from the given parameters (if any).
                ///
                trobject get(const string_t& id, const string_t& params = string_t()) const;

                ///
                /// \brief get the IDs of all registered objects.
                ///
                strings_t ids() const;

                ///
                /// \brief get the IDs of the registered objects matching the ID regex.
                ///
                strings_t ids(const std::regex& id_regex) const;

                ///
                /// \brief get the descriptions of all registered objects.
                ///
                strings_t descriptions() const;

                ///
                /// \brief get the default configurations of all registered objects.
                ///
                strings_t configs() const;

	private:

                struct proto_t
                {
                        tmaker          m_maker;
                        string_t        m_description;
                };
                using protos_t = std::map<string_t, proto_t>;

                // attributes
                protos_t                m_protos;       ///< registered object instances
        };

        template <typename tobject> template<typename tobject_impl>
        bool factory_t<tobject>::add(const string_t& id, const string_t& description)
        {
                const auto maker = [] (const string_t& config) { return std::make_unique<tobject_impl>(config); };
                return m_protos.emplace(id, proto_t{maker, description}).second;
        }

        template <typename tobject>
        bool factory_t<tobject>::has(const string_t& id) const
        {
                return m_protos.find(id) != m_protos.end();
        }

        template <typename tobject>
        typename factory_t<tobject>::trobject factory_t<tobject>::get(const string_t& id, const string_t& configuration) const
        {
                const auto it = m_protos.find(id);
                if (it == m_protos.end())
                {
                        throw std::runtime_error(
                                "invalid object id <" + id + "> of type <" + typeid(tobject).name() + ">!");
                }
                return it->second.m_maker(configuration);
        }

        template <typename tobject>
        strings_t factory_t<tobject>::ids() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.first);
                }
                return ret;
        }

        template <typename tobject>
        strings_t factory_t<tobject>::ids(const std::regex& id_regex) const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        if (std::regex_match(proto.first, id_regex))
                        {
                                ret.push_back(proto.first);
                        }
                }
                return ret;
        }

        template <typename tobject>
        strings_t factory_t<tobject>::descriptions() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.second.m_description);
                }
                return ret;
        }

        template <typename tobject>
        strings_t factory_t<tobject>::configs() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.second.m_maker(string_t())->config());
                }
                return ret;
        }
}
