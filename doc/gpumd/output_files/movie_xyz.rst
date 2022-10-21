.. _movie_xyz:
.. index::
   single: movie.xyz (output file)

``movie.xyz``
=============

This file contains the positions (trajectory) of the atoms sampled at a given frequency.
It is geenrated when invoking the :ref:`dump_position keyword <kw_dump_position>`.

File format
-----------
This file is written in `extended xyz format <https://en.wikipedia.org/wiki/XYZ_file_format>`_.
Specifically, it reads::
  
    number_of_atoms
    frame_label
    type_1 x1 y1 z1
    ...
    number_of_atoms
    frame_label
    type_1 x1 y1 z1
    ...

* The first line: :attr:`number_of_atoms` is the number of atoms for one frame.
* The second line: :attr:`frame_label` is the frame label (starting from 0) within a run.
* The third line: :attr:`type_1` is the atom type for atom 1 and :attr:`x1 y1 z1` are the position components (in units of Ångstrom) for this atom.
* Then there are :attr:`number_of_atoms - 1` additional lines for the current frame.
* Then the pattern above is repeated.
