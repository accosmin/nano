#pragma once

#include <map>
#include <stdexcept>
#include "clonable.h"

namespace nano
{
        ///
        /// \brief manage objects of similar type.
        ///
        template <class tobject>
        class manager_t
	{
        public:

                using trobject = std::unique_ptr<tobject>;
                using tmaker = std::function<trobject(const string_t&)>;

                ///
                /// \brief register a new object with the given ID.
                ///
                bool add(const string_t& id, const string_t& description, const tmaker& maker);

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

        template <class tobject>
        bool manager_t<tobject>::add(const string_t& id, const string_t& description, const tmaker& maker)
        {
                return m_protos.emplace(id, proto_t{maker, description}).second;
        }

        template <class tobject>
        bool manager_t<tobject>::has(const string_t& id) const
        {
                return m_protos.find(id) != m_protos.end();
        }

        template <class tobject>
        typename manager_t<tobject>::trobject manager_t<tobject>::get(const string_t& id, const string_t& configuration) const
        {
                const auto it = m_protos.find(id);
                if (it == m_protos.end())
                {
                        throw std::runtime_error(
                                "invalid object id <" + id + "> of type <" + typeid(tobject).name() + ">!");
                }
                return it->second.m_maker(configuration);
        }

        template <class tobject>
        strings_t manager_t<tobject>::ids() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.first);
                }
                return ret;
        }

        template <class tobject>
        strings_t manager_t<tobject>::descriptions() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.second.m_description);
                }
                return ret;
        }

        template <class tobject>
        strings_t manager_t<tobject>::configs() const
        {
                strings_t ret;
                for (const auto& proto : m_protos)
                {
                        ret.push_back(proto.second.m_maker(string_t())->config());
                }
                return ret;
        }
}
