#pragma once

#include <map>
#include <regex>
#include <memory>
#include "stringi.h"
#include <stdexcept>
#include <functional>

namespace nano
{
        ///
        /// \brief create objects of similar type that implement the configurable_t interface.
        ///
        template <typename tobject, typename... targs>
        class factory_t
	{
        public:
                using trobject = std::unique_ptr<tobject>;
                using tmaker = std::function<trobject(const targs&...)>;

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
                /// \brief retrieve the object with the given ID.
                ///
                trobject get(const string_t& id, const targs&...) const;

                ///
                /// \brief get the IDs of the registered objects matching the ID regex.
                ///
                strings_t ids(const std::regex& id_regex = std::regex(".+")) const;

                ///
                /// \brief returns the number of registered objects.
                ///
                size_t size() const { return m_protos.size(); }

                ///
                /// \brief get the descriptions of all registered objects.
                ///
                strings_t descriptions() const;

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

        template <typename tobject, typename... targs> template <typename tobject_impl>
        bool factory_t<tobject, targs>::add(const string_t& id, const string_t& description)
        {
                static_assert(std::is_base_of<tobject, tobject_impl>::value, "");
                const auto maker = [] (const targs&... args)
                {
                        return std::make_unique<tobject_impl>(std::forward<targs>(args)...);
                };
                return m_protos.emplace(id, proto_t{maker, description}).second;
        }

        template <typename tobject, typename... targs>
        bool factory_t<tobject, targs>::has(const string_t& id) const
        {
                return m_protos.find(id) != m_protos.end();
        }

        template <typename tobject, typename... targs>
        typename factory_t<tobject, targs>::trobject factory_t<tobject>::get(const string_t& id, const targs&... args) const
        {
                const auto it = m_protos.find(id);
                if (it == m_protos.end())
                {
                        throw std::runtime_error(
                                "invalid object id <" + id + "> of type <" + typeid(tobject).name() + ">!");
                }
                return it->second.m_maker(std::forward<targs>(args)...);
        }

        template <typename tobject, typename... targs>
        strings_t factory_t<tobject, targs>::ids(const std::regex& id_regex) const
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

        template <typename tobject, typename... targs>
        strings_t factory_t<tobject, targs>::descriptions() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.second.m_description);
                }
                return ret;
        }
}
