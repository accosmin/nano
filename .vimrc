set tabstop=8       " The width of a TAB
set shiftwidth=8    " Indent size
set softtabstop=8   " Sets the number of columns for a TAB
set expandtab       " Expand TABs to spaces
set smarttab        " Make "tab" insert indents instead of tabs at the beginning of a line
set exrc
set secure

syntax on

map <F2> :retab <CR> :wq! <CR>

" Source: http://www.alexeyshmalko.com/2014/using-vim-as-c-cpp-ide/
set colorcolumn=120
highlight ColorColumn ctermbg=darkgray
set path+=./src
set path+=./src/cortex
set path+=./src/thread
set path+=./src/text
set path+=./src/math
set path+=./src/io
