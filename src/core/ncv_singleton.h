#ifndef NANOCV_SINGLETON_H
#define NANOCV_SINGLETON_H

#include <memory>
#include <mutex>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Singleton.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        template
        <
                class tobject
        >
        class singleton
	{
        public:

                typedef tobject                                 this_object;
                typedef std::unique_ptr<this_object>            this_instance_t;
                typedef std::once_flag                          this_mutex_t;

                // Access the only instance
                static this_object& instance()
                {
                        std::call_once(m_once_flag, []()
                        {
                                m_instance.reset(new this_object());
                        });
                        return *m_instance.get();
                }

                // Destructor
                virtual ~singleton() {}

        protected:

                // Constructor
                singleton() {}

        private:

                // Disable copying
                singleton(const singleton& other) = delete;
                singleton(singleton&& other) = delete;
                singleton& operator=(const singleton& other) = delete;
		
	private:

                // Attributes
                static this_instance_t  m_instance;
                static this_mutex_t     m_once_flag;
	};
        
        template <class tobject>
        typename singleton<tobject>::this_instance_t    singleton<tobject>::m_instance = nullptr;
        
        template <class tobject>
        typename singleton<tobject>::this_mutex_t       singleton<tobject>::m_once_flag;
}

#endif // NANOCV_SINGLETON_H

