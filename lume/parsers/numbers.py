# ------ Number parsing ------
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isbool(x):
    z = x.strip().strip(".").upper()
    if z in ["T", "TRUE", "F", "FALSE"]:
        return True
    else:
        return False


def try_int(x):
    if x == int(x):
        return int(x)
    else:
        return x


def try_bool(x):
    z = x.strip().strip(".").upper()
    if z in ["T", "TRUE"]:
        return True
    elif z in ["F", "FALSE"]:
        return False
    else:
        return x


# Simple function to try casting to a float, bool, or int
def number(x):
    z = x.replace("D", "E")  # Some floating numbers use D
    if isfloat(z):
        val = try_int(float(z))
    elif isbool(x):
        val = try_bool(x)
    else:
        # must be a string. Strip quotes.
        val = x.strip().strip("'").strip('"')
    return val
