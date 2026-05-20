r"""
Base class for EDA Problems.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from enum import Enum


class EDA_BENCH(str, Enum):
    """Define the EDA benchmark names as an enumeration."""
    DAC2012 = "DAC2012"
    ICCAD2014 = "ICCAD2014"
    ICCAD2015_OT = "ICCAD2015_OT"
    ICCAD2015_HS = "ICCAD2015_HS"
    ISPD2005 = "ISPD2005"
    ISPD2005FREE = "ISPD2005Free"
    ISPD2015 = "ISPD2015"
    ISPD2019 = "ISPD2019"
    MMS = "MMS"