#pragma once

namespace cortex
{
        template
        <
                class tobject
        >
        class singleton_t
        {
        public:
                ///
                /// \brief acccess the only instance
                ///
                static tobject& instance()
                {
                        static tobject the_instance;
                        return the_instance;
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
        };
}
