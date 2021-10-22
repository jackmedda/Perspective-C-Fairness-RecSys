#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Container: abstract classes
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging
from abc import ABCMeta

import numpy as np
from six import with_metaclass

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class ObjectUtilMixin(object):
    """
    Methods that are commonly used in data containers and recommenders for
    handling object.
    """

    def _set_object_info(self, data):
        """
        import object meta information of input data to recommenders

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.BaseData` class
        """
        if not isinstance(data, BaseData):
            raise TypeError("input data must data.BaseData class")

        self.n_otypes = data.n_otypes
        self.n_objects = data.n_objects
        self.eid = data.eid
        self.iid = data.iid

    def to_eid(self, otype, iid):
        """
        convert an internal id to the corresponding external id

        Parameters
        ----------
        otype : int
            object type
        iid : int
            an internal id

        Returns
        -------
        eid : int
            the corresponding external id

        Raises
        ------
        ValueError
            an internal id is out of range
        """
        try:
            return self.eid[otype][iid]
        except IndexError:
            raise ValueError("Illegal internal id")

    def to_iid(self, otype, eid):
        """
        convert an external id to the corresponding internal id

        Parameters
        ----------
        otype : int
            object type
        eid : int
            an external id

        Returns
        -------
        iid : int
            the corresponding internal id

        Raises
        ------
        ValueError
            an external id is out of range
        """
        try:
            return self.iid[otype][eid]
        except KeyError:
            raise ValueError("Illegal external id")

    @staticmethod
    def _gen_id(event):
        """
        Generate a conversion map between internal and external ids

        Parameter
        ---------
        event : array_like
            array contains all objects of the specific type

        Returns
        -------
        n_objects : int
            the number of unique objects
        eid : array, shape=(variable,)
            map from internal id to external id
        iid : dict
            map from external id to internal id
        """

        eid = np.unique(event)
        iid = {k: i for (i, k) in enumerate(eid)}

        return len(eid), eid, iid

    @staticmethod
    def _gen_id_substitution_table(orig_obj, sub_index):
        """
        A sub-array of objects are generated by selecting some objects in an
        original array.  This routine returns the table to the original object
        iid to the corresponding iid in a sub-array.

        Parameters
        ----------
        orig_obj : array, shape=(n_objects,)
            Original array of objects.  The iid-th element of this array is its
            corresponding eid.
        sub_index : array, dtype=int, shape=(n_subobjects,)
            Array of indexes included in a sub-array.

        Returns
        -------
        table : array, dtype=int, shape(n_objects,)
            A  substitution table to original eid indexes to the indexes of
            a sub-object array.
        """
        table = np.tile(-1, orig_obj.shape)
        table[sub_index] = np.arange(sub_index.shape[0])

        return table


class BaseData(with_metaclass(ABCMeta, ObjectUtilMixin, object)):
    """
    Abstract class for data container

    Instances of this class contain only information about objects.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)

    Attributes
    ----------
    n_objects : array_like, shape=(n_otypes), dtype=int
        the number of different objects in each type.  the first and the
        second types of objects are referred by the keywords, ``user`` or
        ``item``.
    eid : array_like, shape=(n_otypes,), dtype=(array_like)
        id[i] is a vector of external ids. the j-th element of the array is the
        external id that corresponds to the object with internal id j.
    iid : dictionary
        id[i] is a dictionary for internal ids whose object type is i. the
        value for the key 'j' contains the internal id of the object whose
        external id is j.
    feature : array_like, shape=(n_otypes), dtype=array_like
        i-the element contains the array of features for i-th object types,
        whose shape is (n_object[i], variable). j-th row of the i-th array
        contains a feature for the object whose internal id is j.

    Raises
    ------
    ValueError
        if n_otypes < 1

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2):

        self.n_otypes = 0
        self.n_objects = None
        self.eid = None
        self.iid = None

        if n_otypes < 1:
            raise ValueError("n_otypes must be >= 1")
        self.n_otypes = n_otypes
        self.n_objects = np.zeros(self.n_otypes, dtype=int)
        self.eid = np.tile(None, self.n_otypes)
        self.iid = np.tile(None, self.n_otypes)
        self.feature = np.tile(None, self.n_otypes)

    def set_feature(self, otype, eid, feature):
        """
        Set object feature
        
        Parameters
        ----------
        otype : int
            target object type
        eid : array_like, shape=(n_objects,)
            external ids of the corresponding object features
        feature : array_like
            array of object feature
        """
        iid = self.iid[otype]
        index = np.repeat(len(eid), len(iid))
        for i, j in enumerate(eid):
            if j in iid:
                index[iid[j]] = i
        self.feature[otype] = feature[index].copy()


# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system ---------------------------------------------------------
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
