#ifndef NANOCV_UNROLL_H
#define NANOCV_UNROLL_H

namespace ncv
{        
        ///
        /// \brief unroll the given loop 
        ///
        template
        <
                typename toperator
        >
        void unroll(int size, const toperator& op)
        {
                const int size4 = size - (size & 3);
                
                for (int i = 0; i < size4; i += 4)
                {
                        op(i + 0);
                        op(i + 1);
                        op(i + 2);
                        op(i + 3);
                }
                
                if (size > size4)
                {
                        for (int i = size4; i < size; i ++)
                        {
                                op(i);
                        }
                }
        }
        
        ///
        /// \brief unroll the given fixed-size loop 
        ///
        template
        <
                typename toperator,
                int tsize
        >
        void unroll(const toperator& op)
        {
                const int tsize4 = tsize - (tsize & 3);
                
                for (int i = 0; i < tsize4; i += 4)
                {
                        op(i + 0);
                        op(i + 1);
                        op(i + 2);
                        op(i + 3);
                }
                
                if (tsize > tsize4)
                {
                        for (int i = tsize4; i < tsize; i ++)
                        {
                                op(i);
                        }
                }
        }
}

#endif // NANOCV_UNROLL_H

