#pragma once

#include <map>
#include <stdexcept>
#include "clonable.hpp"

namespace nano
{
        ///
        /// \brief manage objects of similar type.
        ///
        template <class tobject>
        class manager_t
	{
        public:

                using trobject = std::shared_ptr<tobject>;

                ///
                /// \brief add a new object with the given ID
                ///
                bool add(const string_t& id, const tobject& proto)
                {
                        return m_protos.emplace(id, proto.clone()).second;
                }

                ///
                /// \brief check if an objects was registered with the given ID
                ///
                bool has(const string_t& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }

                ///
                /// \brief retrieve the object with the given ID
                ///
                trobject get(const string_t& id) const
                {
                        return get_it(id)->clone();
                }

                ///
                /// \brief retrieve the object with the given ID, constructed from the given parameters
                ///
                trobject get(const string_t& id, const string_t& params) const
                {
                        return get_it(id)->clone(params);
                }

                ///
                /// \brief get the IDs of all registered objects
                ///
                strings_t ids() const
                {
                        return collect<string_t>([] (const auto& it) { return it.first; });
                }

                ///
                /// \brief get the descriptions of all registered objects
                ///
                strings_t descriptions() const
                {
                        return collect<string_t>([] (const auto& it) { return it.second->description(); });
                }

                ///
                /// \brief get the configurations of all registered objects
                ///
                strings_t configs() const
                {
                        return collect<string_t>([] (const auto& it) { return it.second->config(); });
                }

        private:

                template <typename treturn, typename tfunctor>
                auto collect(const tfunctor& fun) const
                {
                        std::vector<treturn> result;
                        for (const auto& proto : m_protos)
                        {
                                result.push_back(fun(proto));
                        }
			return result;
                }

                const trobject& get_it(const string_t& id) const
                {
                        const auto it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::runtime_error(
                                        "invalid object id <" + id + "> of type <" + typeid(tobject).name() + ">!");
                        }
                        return it->second;
                }

	private:

                // attributes
                std::map<string_t, trobject> m_protos;       ///< registered object instances
        };
}
