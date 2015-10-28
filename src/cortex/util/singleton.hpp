#pragma once

#include <memory>
#include <mutex>

namespace cortex
{
        template
        <
                class tobject
        >
        class singleton_t
	{
        public:

                using this_object = tobject;
                using this_mutex_t = std::once_flag;
                using this_instance_t = std::unique_ptr<this_object>;

                ///
                /// \brief acccess the only instance
                ///
                static this_object& instance()
                {
                        std::call_once(m_once_flag, []()
                        {
                                singleton_t::m_instance.reset(new this_object);
                        });
                        return *m_instance;
                }

        protected:

                ///
                /// \brief constructor
                ///
                singleton_t() {}

                ///
                /// \brief disable copying
                ///
                singleton_t(const singleton_t&) = delete;
                singleton_t& operator=(const singleton_t&) = delete;

                ///
                /// \brief destructor
                ///
                virtual ~singleton_t() {}
		
	private:

                // attributes
                static this_instance_t  m_instance;
                static this_mutex_t     m_once_flag;
	};
        
        template <class tobject>
        typename singleton_t<tobject>::this_instance_t    singleton_t<tobject>::m_instance = nullptr;
        
        template <class tobject>
        typename singleton_t<tobject>::this_mutex_t       singleton_t<tobject>::m_once_flag;
}
