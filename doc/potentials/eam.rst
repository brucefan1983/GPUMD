.. _eam:
.. index::
   single: Embedded atom method

Embedded atom method
====================

:program:`GPUMD` suppports two different analytical forms of embedded atom method (:term:`EAM`) potentials.
Using the form by Zhou *et al.* can simulate alloys with up to 10 atom types [Zhou2004]_, while using the form of Dai *et al.* the implementation only applies to systems with a single atom type [Dai2006]_.

Potential form
--------------

General form
^^^^^^^^^^^^

The site potential energy is

.. math::
   
   U_i = \frac{1}{2} \sum_{j\neq i} \phi(r_{ij}) + F (\rho_i).

Here, the part with :math:`\phi(r_{ij})` is a pairwise potential and :math:`F (\rho_i)` is the embedding potential, which depends on the electron density :math:`\rho_i` at site :math:`i`.
The many-body part of the EAM potential comes from the embedding potential.  

The density :math:`F (\rho_i)` is contributed by the neighbors of :math:`i`:

.. math::
   
   \rho_i = \sum_{j\neq i} f(r_{ij}).

Therefore, the form of an :term:`EAM` potential is completely determined by the three functions: :math:`\phi`, :math:`f`, and :math:`F`.

Version from [Zhou2004]_
^^^^^^^^^^^^^^^^^^^^^^^^

The pair potential between two atoms of the same type :math:`a` is

.. math::
   
   \phi^{aa}(r) = \frac{ A^a \exp[-\alpha(r/r_e^a-1)] } { 1+(r/r_e^a-\kappa^a)^{20} } -
   \frac{ B^a \exp[-\beta(r/r_e^a-1)] } { 1+(r/r_e^a-\lambda^a)^{20} }.

The contribution of the electron density from an atom of type :math:`a` is

.. math::

   f^a(r) = \frac{ f_e^a \exp[-\beta(r/r_e^a-1)] } { 1+(r/r_e^a-\lambda^a)^{20} }.

The pair potential between two atoms of different types :math:`a` and :math:`b` is then constructed as

.. math::
   
   \phi^{ab}(r) = \frac{1}{2}
   \left[
   \frac{ f^b(r) } { f^a(r) } \phi^{aa}(r) + \frac{ f^a(r) } { f^b(r) } \phi^{bb}(r)
   \right].

The embedding energy function is piecewise:

.. math::
   
   F(\rho) = \begin{cases}
   \sum_{i=0}^3 F_{ni} \left( \frac{\rho}{\rho_n}-1\right)^i
   & \rho < 0.85\rho_e \\
   \sum_{i=0}^3 F_{i} \left( \frac{\rho}{\rho_e}-1\right)^i
   & 0.85\rho_e \leq \rho < 1.15\rho_e \\
   F_{e} \left[ 1- \ln \left(\frac{\rho}{\rho_s}\right)^{\eta}\right] \left(\frac{\rho}{\rho_s}\right)^{\eta}
   & \rho \geq 1.15\rho_e
   \end{cases}

Version from [Dai2006]_
^^^^^^^^^^^^^^^^^^^^^^^

This is a very simple :term:`EAM`-type potential, which is an extension of the Finnis-Sinclair potential.
The function for the pair potential is

.. math::
   
   \phi(r) = \begin{cases}
   (r-c)^2 \sum_{n=0}^4 c_n r^n  & r \leq c \\
   0                             & r > c
   \end{cases}

The function for the density is

.. math::
   
   f(r) =
   \begin{cases}
   (r-d)^2 + B^2 (r-d)^4  & r \leq d \\
   0                      & r > d
   \end{cases}

The function for the embedding energy is

.. math::

   F(\rho) = - A \rho^{1/2}.

File format
-----------

The potential file for the version from [Zhou2004]_ reads::

  eam_zhou_2004 num_types
  r_e f_e rho_e rho_s alpha beta A B kappa lambda F_n0 F_n1 F_n2 F_n3 F_0 F_1 F_2 F_3 eta F_e cutoff
  
There are :attr:`num_types` rows of parameters but here we have only written a single row above.
The order of the rows should be consistent with the atoms types you defined in the :ref:`simulation model file <model_xyz>` and specified by the :attr:`potential` keyword in the :ref:`run.in file <run_in>`.

The last parameter :attr:`cutoff` is the cutoff distance, which is not intrinsic to the model.
The order of the parameters is the same as in Table III of [Zhou2004]_.
For multi-component systems, :program:`GPUMD` will use the largest cutoff for every atom type.

The potential file for the version from [Dai2006]_ reads::

  eam_dai_2006 1
  A d c c_0 c_1 c_2 c_3 c_4 B
