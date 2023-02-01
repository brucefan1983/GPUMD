.. _kw_type:
.. index::
   single: prediction (keyword in nep.in)

:attr:`prediction`
==================

This keyword instructs :program:`nep` to evaluate a model against a set of structures without starting an optimization.
This requires a ``nep.txt`` file to be present.
Note that only the structures in ``train.xyz`` are included in the prediction.
The syntax is::

  prediction <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===========================
Value  Mode 
-----  ---------------------------
0      optimization mode (default)
1      prediction mode
=====  ===========================
