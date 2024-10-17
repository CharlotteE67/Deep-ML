# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/16 19:42
import math

import numpy as np

count = eval(input())
# print(sum(count["user1"].values()))
ans = {user: {"entropy": round(sum([-v/sum(value.values()) * math.log2(v/sum(value.values())) for k, v in value.items()]), 3)} for user, value in count.items()}
# print(ans)
print(str(ans).replace("'", "\""))

"""
{"user1": {"sports": 10, "tech": 20, "en": 30}, "user2": {"sports": 20, "tech": 30, "en": 10}, "user3": {"sports": 30, "tech": 10, "en": 20}}
"""