#pragma once

#include <string>
#include <memory>

namespace ncv
{
        ///
        /// the clonable interface to be used with a manager:
        ///      ::make(const tstring&)         - create a new object (with the given configuration)
        ///      ::clone()                      - create a copy of the current object
        ///      ::configuration()              - parametrization
        ///      ::description()                - short description (configuration included)
        ///
        template
        <
                typename tobject,
                typename tstring = std::string
        >
        class clonable_t
        {        
        public:

                typedef tobject                         object_t;
                typedef tstring                         string_t;
                typedef std::shared_ptr<tobject>        robject_t;

                ///
                /// \brief constructor
                ///
                clonable_t(const tstring& configuration)
                        :       m_configuration(configuration)
                {
                }

                ///
                /// \brief create an object clone
                ///
                virtual robject_t make(const tstring& configuration) const = 0;
                virtual robject_t clone() const = 0;
                
                ///
                /// \brief describe the object
                ///
                tstring configuration() const { return m_configuration; }
                virtual tstring description() const = 0;

        protected:

                // attributes
                tstring         m_configuration;
        };

        #define NANOCV_MAKE_CLONABLE(base_class, description_text) \
                virtual robject_t make(const string_t& configuration) const override \
                { \
                        return robject_t(new base_class(configuration)); \
                } \
                virtual robject_t clone() const override \
                { \
                        return robject_t(new base_class(*this)); \
                } \
                virtual string_t description() const override \
                { \
                        return description_text; \
                }
}
