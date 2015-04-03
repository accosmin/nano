#pragma once

#include "noncopyable.hpp"
#include <memory>
#include <mutex>

namespace ncv
{
        template
        <
                class tobject
        >
        class singleton_t : private noncopyable_t
	{
        public:

                typedef tobject                                 this_object;
                typedef std::unique_ptr<this_object>            this_instance_t;
                typedef std::once_flag                          this_mutex_t;

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
