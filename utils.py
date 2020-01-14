#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:55:03 2020

@author: molano
"""
import hashlib


def sorted_str_from_dict(d):
    '''Creates a string of the key, value pairs in dict d, sorted by key.
    Sorting is required to 'uniquify' any set of hyperparameter settings
    (because two different orderings of the same hyperparameter settings
    are assumed to result in identical model and fitting behavior).

    Args:
        d: dict of hyperparameters.

    Returns:
        string of key, value pairs, sorted by key.

    by Matt Golub, August 2018.
    Please direct correspondence to mgolub@stanford.edu.
    recurrent-whisperer
    '''
    sorted_keys = sorted(d.keys())
    n_keys = len(sorted_keys)

    str_items = ['{']
    key_idx = 1
    for key in sorted_keys:
        val = d[key]

        if isinstance(val, dict):
            str_val = sorted_str_from_dict(val)
        else:
            str_val = str(val)

        new_entry = (str(key) + ': ' + str_val)
        str_items.append(new_entry)

        if key_idx < n_keys:
            str_items.append(', ')

        key_idx += 1

    str_items.append('}')

    return ''.join(str_items)


def generate_hash(hps):
    '''Generates a hash from a unique string representation of the
    hyperparameters in hps.

    Args:
        hps: dict of hyperparameter names and settings as keys and values.

    Returns:
        string containing 512-bit hash in hexadecimal representation.

    by Matt Golub, August 2018.
    Please direct correspondence to mgolub@stanford.edu.
    recurrent-whisperer
    '''
    str_to_hash = sorted_str_from_dict(hps)

    # Generate the hash for that string
    h = hashlib.new('md5')
    str_to_hash = str_to_hash.encode('utf-8')
    h.update(str_to_hash)
    hps_hash = h.hexdigest()

    return hps_hash


if __name__ == '__main__':
    hps = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    print(generate_hash(hps))
    hps = {'d': 4, 'b': 2, 'c': 3, 'a': 1}
    print(generate_hash(hps))

