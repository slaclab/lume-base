from numbers import Number

import numpy as np

from .numbers import isfloat, number

# Parsers
# -------


def clean_namelist_key_value(line):
    """
    Cleans up a namelist "key = value line"

    Removes all spaces from keys

    """
    z = line.split("=")
    key = z[0].strip().replace(" ", "")
    value = "".join(z[1:])
    return f"{key} = {value}"


def unroll_namelist_line(line, commentchar="!", condense=False):
    """
    Unrolls namelist lines. Looks for vectors, or multiple keys per line.
    """
    lines = []
    # Look for comments
    x = line.strip().strip(",").split(commentchar)
    if len(x) == 1:
        # No comments
        x = x[0].strip()
    else:
        # Unroll comment first
        comment = "".join(x[1:])
        if not condense:
            lines.append(commentchar + comment)
        x = x[0].strip()
    if x == "":
        pass
    elif x[0] == "&" or x[0] == "/":
        # This is namelist control. Write.
        lines.append(x)
    else:
        # Content line. Should contain =
        # unroll.
        # Check for multiple keys per line, or vectors.
        # TODO: handle both
        n_keys = len(x.split("="))
        if n_keys == 2:
            # Single key
            lines.append(clean_namelist_key_value(x))
        elif n_keys > 2:
            for y in x.strip(",").split(","):
                lines.append(clean_namelist_key_value(y))

    return lines


def parse_simple_namelist(filePath, commentchar="!", condense=False):
    """
    Unrolls namelist style file. Returns lines.
    makes keys lower case

    Example:

    &my_namelist

        x=1, YY  = 4 ! this is a comment:
    /

    unrolls to:
    &my_namelist
    ! this is a comment
        x = 1
        yy = 4
    /

    """

    lines = []
    with open(filePath) as f:
        for line in f:
            ulines = unroll_namelist_line(
                line, commentchar=commentchar, condense=condense
            )
            lines = lines + ulines

    return lines


def parse_unrolled_namelist(unrolled_lines, end="/", commentchar="!"):
    """
    Parses an unrolled namelist into list of (name, dict)

    Returns tuple:
        names: list of group names
        dicts: lists of dicts of data corresponding to these groups

    """
    names = []
    dicts = []
    n = {}
    for line in unrolled_lines:
        # Ignore these
        if line[0] == "1" or line == end or line[0] == commentchar:
            continue
        if line[0] == "&":
            # New namelist
            name = line[1:]
            names.append(name)
            # point to current namelist
            n = {}
            dicts.append(n)
            continue
        # content line
        key, val = line.split("=")

        # look for vector
        vals = val.split()
        if len(vals) == 1:
            val = number(vals[0])
        else:
            if isfloat(vals[0].replace(",", " ")):
                # Vector. Remove commas
                val = [number(z) for z in val.replace(",", " ").split()]
            else:
                # This is just a string. Just strip
                val = val.strip()
        n[key.strip()] = val

    return names, dicts


# Writers
# -------


def namelist_lines(namelist_dict, name, end="/", strip_strings=False):
    """
    Converts namelist dict to output lines, for writing to file.

    Only allow scalars or lists.

    Do not allow np arrays or any other types from simplicity.
    """
    lines = []
    lines.append("&" + name)
    # parse

    for key, value in namelist_dict.items():
        # if type(value) == type(1) or type(value) == type(1.): # numbers

        if isinstance(value, Number):  # numbers
            line = key + " = " + str(value)
        elif isinstance(value, (list, np.ndarray)):  # lists or np arrays
            liststr = ""
            for item in value:
                liststr += str(item) + " "
            line = key + " = " + liststr
        elif isinstance(value, str):  # strings
            value.strip("''")
            if not strip_strings:
                value = "'" + value.strip("''") + "'"  # input may need apostrophes
            line = key + " = " + value

        elif bool(value) == value:
            line = key + " = " + str(value)
        else:
            # print 'skipped: key, value = ', key, value
            raise ValueError(
                f"Problem writing input key: {key}, value: {value}, type: {type(value)}"
            )

        lines.append(line)

    lines.append(end)
    return lines
