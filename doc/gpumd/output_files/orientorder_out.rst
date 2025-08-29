.. _orientorder_out:
.. index::
   single: orientorder.out (output file)

``orientorder.out``
====================

This file stores the **Steinhardt order parameters**.  
It is automatically generated when the :ref:`compute_orientorder keyword <kw_compute_orientorder>` is invoked.

File format
-----------

The data are written in a repeating block structure:

- **First line**: the current simulation step.  
- **Second line**: column headers indicating the computed degrees.  
  For example, if degrees 4 and 6 are calculated together with their :math:`w_l` versions, the column names will be:  
  ``ql4 ql6 wl4 wl6``  
- **Subsequent lines**: per-atom order parameter values for the given step.
