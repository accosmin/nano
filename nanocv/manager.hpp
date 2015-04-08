#pragma once

#include "clonable.hpp"
#include "singleton.hpp"
#include <vector>
#include <map>
#include <stdexcept>

namespace ncv
{
        ///
        /// \brief manager: used to manage different object types associated with ID strings.
        /// hint: use register_object<base, derived> to register objects to the manager.
        ///
        template
        <
                class tobject,
                typename tstring = std::string
        >
        class manager_t : public singleton_t<manager_t<tobject, tstring> >
	{
        public:

                typedef tstring                                                 string_t;
                typedef typename clonable_t<tobject, tstring>::robject_t        robject_t;

                // manage prototypes
                bool add(const tstring& id, const tobject& proto)
                {
                        return _add(id, proto);
                }
                bool has(const tstring& id) const
                {
                        return _has(id);
                }
                robject_t get(const tstring& id) const
                {
                        return _get(id);
                }
                robject_t get(const tstring& id, const tstring& params) const
                {
                        return _get(id, params);
                }

                // access functions
                std::vector<tstring> ids() const { return _ids(); }
                std::vector<tstring> descriptions() const { return _descriptions(); }

        private:

                // object prototypes
                typedef std::map<tstring, robject_t>       	protos_t;
                typedef typename protos_t::const_iterator       protos_const_it;
                typedef typename protos_t::iterator		protos_it;
                
        private:
                
                bool _add(const tstring& id, const tobject& proto)
                {
                        return m_protos.insert(typename protos_t::value_type(id, proto.clone())).second;
                }

                bool _has(const tstring& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }

                robject_t _get(const tstring& id) const
                {
                        const protos_const_it it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::runtime_error("invalid object id <" + id + ">!");
                        }
                        return it->second->clone();
                }

                robject_t _get(const tstring& id, const tstring& params) const
                {
                        const protos_const_it it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::runtime_error("invalid object id <" + id + ">!");
                        }
                        return it->second->make(params);
                }

		template <typename TFunctor>
                std::vector<tstring> _collect(const TFunctor& fun) const
		{
                        std::vector<tstring> result;
                        for (protos_const_it it = m_protos.cbegin(); it != m_protos.cend(); ++ it)
                        {
                                result.push_back(fun(it));
                        }

			return result;
		}

                std::vector<tstring> _ids() const
                {
			return _collect([] (const protos_const_it& it) { return it->first; });
                }

                std::vector<tstring> _descriptions() const
		{
                        return _collect([] (const protos_const_it& it) { return it->second->description(); });
		}
		
	private:

                // attributes
                protos_t                m_protos;
        };

        ///
        /// \brief register a type tderived to the tbase manager
        ///
        template
        <
                class tbase,
                class tderived
        >
        struct register_object
        {
                register_object(const typename tbase::string_t& id)
                {
                        manager_t<tbase>::instance().add(id, tderived());
                }
        };
}
