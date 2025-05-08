from .LODA import LODA
from .xStream import xStream
from .RSHash import RSHASH
from .SDOs import SDOs
from .SWKNN import SWKNN
from .LODASALMON import LODASALMON

##### Add your model to this list to make it available for use ####
# check if python version is greater than 3.9
import sys
if sys.version_info >= (3, 9):
    from .OIF import OIF
