#pragma once

namespace ncv
{
        class noncopyable_t
        {
        public:

                ///
                /// \brief constructor
                ///
                noncopyable_t() {}

                ///
                /// \brief destructor
                ///
                virtual ~noncopyable_t() {}

        private:

                ///
                /// \brief disable copying
                ///
                noncopyable_t(const noncopyable_t& other) = delete;

                ///
                /// \brief disable copying
                ///
                noncopyable_t(noncopyable_t&& other) = delete;

                ///
                /// \brief disable copying
                ///
                noncopyable_t& operator=(const noncopyable_t& other) = delete;
        };
}
