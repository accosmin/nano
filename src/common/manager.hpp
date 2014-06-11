#ifndef NANOCV_MANAGER_H
#define NANOCV_MANAGER_H

#include "clonable.hpp"
#include "singleton.hpp"
#include <vector>
#include <map>
#include <stdexcept>

namespace ncv
{
        ///
        /// manager: used to manage different object types associated with ID strings.
        /// hint: use register_object<base, derived> to register objects to the manager.
        ///
        template
        <
                class tobject
        >
        class manager_t : public singleton_t<manager_t<tobject> >
	{
        public:

                typedef typename clonable_t<tobject>::robject_t         robject_t;

                // manage prototypes
                bool add(const std::string& id, const tobject& proto)
                {
                        return _add(id, proto);
                }
                bool has(const std::string& id) const
                {
                        return _has(id);
                }
                robject_t get(const std::string& id) const
                {
                        return _get(id);
                }
                robject_t get(const std::string& id, const std::string& params) const
                {
                        return _get(id, params);
                }

                // access functions
                std::vector<std::string> ids() const { return _ids(); }
                std::vector<std::string> descriptions() const { return _descriptions(); }

        private:

                // object prototypes
                typedef std::map<std::string, robject_t>       	protos_t;
                typedef typename protos_t::const_iterator       protos_const_it;
                typedef typename protos_t::iterator		protos_it;
                
        private:
                
                bool _add(const std::string& id, const tobject& proto)
                {
                        return m_protos.insert(typename protos_t::value_type(id, proto.clone())).second;
                }

                bool _has(const std::string& id) const
                {
                        return m_protos.find(id) != m_protos.end();
                }

                robject_t _get(const std::string& id) const
                {
                        const protos_const_it it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::runtime_error("invalid object id <" + id + ">!");
                        }
                        return it->second->clone();
                }

                robject_t _get(const std::string& id, const std::string& params) const
                {
                        const protos_const_it it = m_protos.find(id);
                        if (it == m_protos.end())
                        {
                                throw std::runtime_error("invalid object id <" + id + ">!");
                        }
                        return it->second->make(params);
                }

		template <typename TFunctor>
                std::vector<std::string> _collect(const TFunctor& fun) const
		{
                        std::vector<std::string> result;
                        for (protos_const_it it = m_protos.cbegin(); it != m_protos.cend(); ++ it)
                        {
                                result.push_back(fun(it));
                        }

			return result;
		}

                std::vector<std::string> _ids() const
                {
			return _collect([] (const protos_const_it& it) { return it->first; });
                }

                std::vector<std::string> _descriptions() const
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
                register_object(const std::string& id)
                {
                        manager_t<tbase>::instance().add(id, tderived());
                }
        };
}

#endif // NANOCV_MANAGER_H

