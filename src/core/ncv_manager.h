#ifndef NANOCV_MANAGER_H
#define NANOCV_MANAGER_H

#include "ncv_string.h"
#include "ncv_singleton.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // manager: used to manage different object types associated with ID strings.
        // the clonable interface to be used with a manager:
        //      ::clone(const string_t&)        - create a new object (with the given parameters)
	//      ::name()                        - details the associated ID
        //      ::desc()                        - short description (parameters included)
        // hint: use register_object<base, derived> to register objects to the manager.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject>        robject_t;

                // constructor
                clonable_t(const string_t& name, const string_t& description)
                        :       m_name(name), m_description(description)
                {
                }
        
                // create an object clone
                virtual robject_t clone(const string_t& params) const = 0;
                
                // describe the object
                const string_t& name() const { return m_name; }
                const string_t& description() const { return m_description; }

        private:

                // attributes
                string_t                m_name;
                string_t                m_description;
        };

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
                robject_t get(const string_t& id, const string_t& params) const
                {
                        return _get(id, params);
                }

                // access functions
		strings_t ids() const { return _ids(); }
		strings_t names() const { return _names(); }
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
                        if (m_protos.find(id) == m_protos.end())
                        {
                                m_protos[id] = proto.clone("");
                                return true;
                        }
                        else
                        {
                                return false;
                        }
                }

                //-------------------------------------------------------------------------------------------------

                bool _has(const string_t& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }
                
                //-------------------------------------------------------------------------------------------------

                robject_t _get(const string_t& id, const string_t& params) const
                {
                        const protos_const_it it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::invalid_argument("cannot find the object <" + id + ">!");
                        }
                        return it->second->clone(params);
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

		strings_t _names() const
		{
			return _collect([] (const protos_const_it& it) { return it->second->name(); });
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

