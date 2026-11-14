.. _kw_prediction:
.. index::
   single: prediction (keyword in nep.in)

:attr:`prediction`
==================

Evaluate an existing ``nep.txt`` model on the structures in ``train.xyz``.
This mode requires ``nep.in``, ``nep.txt``, and ``train.xyz``; it does not optimize the model or evaluate ``test.xyz``.
The syntax is::

  prediction <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  ===========================
Value  Mode 
-----  ---------------------------
0      optimization mode (default)
1      prediction mode
=====  ===========================

To evaluate a model, include::

  prediction 1

The predicted and reference quantities are written to the corresponding ``*_train.out`` files listed under :doc:`../output_files/index`.
