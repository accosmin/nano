set tabstop=8       " The width of a TAB
set shiftwidth=8    " Indent size
set softtabstop=8   " Sets the number of columns for a TAB
set expandtab       " Expand TABs to spaces
set smarttab        " Make "tab" insert indents instead of tabs at the beginning of a line
set exrc
set secure

set number
syntax on

" Replace tabs with spaces
map <F2> :retab <CR> :w <CR>

" Toggle whitespace visibility
nmap <F3> :set list!<CR>
set listchars=tab:>-,trail:-

" Toggle between header and implementation
map <F4> :e %:p:s,.h$,.X123X,:s,.cpp$,.h,:s,.X123X$,.cpp,<CR>

" http://www.alexeyshmalko.com/2014/using-vim-as-c-cpp-ide/
set colorcolumn=120
highlight ColorColumn ctermbg=darkgray
set path+=src
set path+=src/cortex
set path+=src/thread
set path+=src/text
set path+=src/math
set path+=src/io
set path+=apps
set path+=test

" Trim trailing whitespaces when saving                                                                                                     
function! StripTrailingWhitespaces()                                                                                                        
        let l = line(".")                                                                                                                       
        let c = col(".")                                                                                                                        
        %s/\s\+$//e                                                                                                                             
        call cursor(l, c)                                                                                                                       
endfunction                                                                                                                                 
autocmd BufWritePre     * :call StripTrailingWhitespaces()  
