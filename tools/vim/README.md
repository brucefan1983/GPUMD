# Vim Syntax Highlighting

Will enable syntax highlighting for the `GPUMD` script syntax in vim. The scripts name with run.in. One can easily add new ones.

## To enable the highlighting

Here is an alternate method for VIM Version 7.2 and later:

1. Create/edit `~/.vimrc` to contain:
```
let mysyntaxfile = "~/.vim/mysyntax.vim"
syntax on
" do not create <TAB> characters
set expandtab
```
2. Create directories `~/.vim/syntax` and `~/.vim/ftdetect`;
3. Create file `~/.vim/mysyntax.vim` to contain:
```
augroup syntax
au  BufNewFile,BufReadPost run.in so ~/.vim/syntax/gpumd.vim
au  BufNewFile,BufReadPost nep.in so ~/.vim/syntax/gpumd.vim
augroup END
```
4. Copy `gpumd.vim` to `~/.vim/syntax/gpumd.vim`
5. Copy `filetype.vim` to `~/.vim/ftdetect/gpumd.vim`

## contact

Ke Xu <twtdq@qq.com>

-----

- updated by Ke Xu, 2025-4-17
- updated by Ke Xu, 2023-5-30
- updated by Ke Xu, 2021

