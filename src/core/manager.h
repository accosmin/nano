#ifndef NANOCV_MANAGER_H
#define NANOCV_MANAGER_H

#include "string.h"
#include "singleton.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // manager: used to manage different object types associated with ID strings.
        // the clonable interface to be used with a manager:
        //      ::clone(const string_t&)        - create a new object (with the given parameters)
        //      ::description()                 - short description (parameters included)
        // hint: use register_object<base, derived> to register objects to the manager.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject> robject_t;

                // create an object clone
                virtual robject_t clone(const string_t& params) const = 0;
                
                // describe the object
                virtual string_t description() const = 0;
        };

        // implements the clonable_t interface
        #define NCV_MAKE_CLONABLE(object_class, base_class, description_str) \
                typedef typename clonable_t<base_class>::robject_t robject_t; \
                \
                virtual robject_t clone(const string_t& params) const \
                { \
                        return robject_t(new object_class(params)); \
                } \
                \
                virtual string_t description() const { return #description_str; }

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

