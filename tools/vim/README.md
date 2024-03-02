=== Vim Syntax Highlighting ===
===============================

Will enable syntax highlighting for the GPUMD script syntax in vim.
The scripts name with run.in or phonon.in.
One can easily add new ones.

=To enable the highlighting:
============================
=Here is an alternate method for VIM Version 7.2 and later:

(0) Create/edit ~/.vimrc to contain:
         syntax on
(1) Create directories ~/.vim/syntax and ~/.vim/ftdetect
(2) Copy gpumd.vim to ~/.vim/syntax/gpumd.vim
(3) Copy filetype.vim to ~/.vim/ftdetect/gpumd.vim

Ke Xu <twtdq@stu.xmu.edu.cn> 2021
