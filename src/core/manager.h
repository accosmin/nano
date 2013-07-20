#ifndef NANOCV_MANAGER_H
#define NANOCV_MANAGER_H

#include "clonable.h"
#include "singleton.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // manager: used to manage different object types associated with ID strings.
        // hint: use register_object<base, derived> to register objects to the manager.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        template
        <
                class tobject
        >
        class manager_t : public singleton_t<manager_t<tobject> >
	{
        public:

                typedef typename clonable_t<tobject>::robject_t         robject_t;

                // manage prototypes
                bool add(const string_t& id, const tobject& proto)
                {
                        return _add(id, proto);
                }
                bool has(const string_t& id) const
                {
                        return _has(id);
                }
                robject_t get(const string_t& id) const
                {
                        return _get(id);
                }
                robject_t get(const string_t& id, const string_t& params) const
                {
                        return _get(id, params);
                }

                // access functions
                strings_t ids() const { return _ids(); }
                strings_t descriptions() const { return _descriptions(); }

        private:

                // object prototypes
                typedef std::map<string_t, robject_t>       	protos_t;
                typedef typename protos_t::const_iterator       protos_const_it;
                typedef typename protos_t::iterator		protos_it;
                
        private:
                
                //-------------------------------------------------------------------------------------------------

                bool _add(const string_t& id, const tobject& proto)
                {
                        return m_protos.insert(typename protos_t::value_type(id, proto.clone(""))).second;
                }

                //-------------------------------------------------------------------------------------------------

                bool _has(const string_t& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }

                //-------------------------------------------------------------------------------------------------

                robject_t _get(const string_t& id) const
                {
                        const protos_const_it it = m_protos.find(id);
                        return it == m_protos.end() ? robject_t() : it->second->clone();
                }
                
                //-------------------------------------------------------------------------------------------------

                robject_t _get(const string_t& id, const string_t& params) const
                {
                        const protos_const_it it = m_protos.find(id);
                        return it == m_protos.end() ? robject_t() : it->second->clone(params);
                }

                //-------------------------------------------------------------------------------------------------

		template <typename TFunctor>
		strings_t _collect(const TFunctor& fun) const
		{
                        strings_t result;
                        for (protos_const_it it = m_protos.cbegin(); it != m_protos.cend(); ++ it)
                        {
                                result.push_back(fun(it));
                        }

			return result;
		}

                //-------------------------------------------------------------------------------------------------

		strings_t _ids() const
                {
			return _collect([] (const protos_const_it& it) { return it->first; });
                }

                //-------------------------------------------------------------------------------------------------

                strings_t _descriptions() const
		{
                        return _collect([] (const protos_const_it& it) { return it->second->description(); });
		}

                //-------------------------------------------------------------------------------------------------
		
	private:

                // attributes
                protos_t                        m_protos;
        };
        
        // register a type tderived to the tbase manager
        template
        <
                class tbase,
                class tderived
        >
        struct register_object
        {
                register_object(const string_t& id)
                {
                        manager_t<tbase>::instance().add(id, tderived());
                }
        };
}

#endif // NANOCV_MANAGER_H

