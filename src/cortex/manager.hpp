#pragma once

#include "clonable.hpp"
#include "singleton.hpp"
#include <vector>
#include <map>
#include <stdexcept>

namespace cortex
{
        ///
        /// \brief manager: used to manage different object types associated with ID strings.
        ///
        template
        <
                class tobject,
                typename tstring = std::string
        >
        class manager_t : public singleton_t<manager_t<tobject, tstring> >
	{
        public:

                typedef typename clonable_t<tobject, tstring>::robject_t        robject_t;

                ///
                /// \brief add a new object with the given ID
                ///
                bool add(const tstring& id, const tobject& proto)
                {
                        return m_protos.emplace(id, proto.clone()).second;
                }

                ///
                /// \brief check if an objects was registered, given its ID
                ///
                bool has(const tstring& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }

                ///
                /// \brief retrieve the object associated with the given ID
                ///
                robject_t get(const tstring& id) const
                {
                        const auto it = m_protos.find(id);
                        assert_it(id, it);
                        return it->second->clone();
                }

                ///
                /// \brief retrieve the object associated with the given ID, constructed from the given parameters
                ///
                robject_t get(const tstring& id, const tstring& params) const
                {
                        const auto it = m_protos.find(id);
                        assert_it(id, it);
                        return it->second->make(params);
                }

                ///
                /// \brief get the IDs of all registered objects
                ///
                std::vector<tstring> ids() const
                {
                        return collect([] (const auto& it) { return it.first; });
                }

                ///
                /// \brief get the descriptions of all registered objects
                ///
                std::vector<tstring> descriptions() const
                {
                        return collect([] (const auto& it) { return it.second->description(); });
                }

        private:

                template <typename tfunctor>
                std::vector<tstring> collect(const tfunctor& fun) const
		{
                        std::vector<tstring> result;
                        for (const auto& proto : m_protos)
                        {
                                result.push_back(fun(proto));
                        }

			return result;
                }

                template <typename titerator>
                void assert_it(const tstring& id, const titerator& it) const
                {
                          if (it == m_protos.end())
                          {
                                  throw std::runtime_error("invalid object id <" + id + ">!");
                          }
                }
		
	private:

                // attributes
                std::map<tstring, robject_t>    m_protos;       ///< registered objects
        };
}
